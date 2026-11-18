const INTAKE_SHEET = 'Shipments';
const TFWW_ACCOUNTS = ['Covenant', 'Ultra Play'];
const HEADERS = ['Origin ZIP', 'TFWW Account', 'Source File', 'Status', 'Review Notes', 'Reviewed Shipment (JSON)', 'TFWW Result', 'Submission ID'];

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Shipment Intake')
    .addItem('Upload Shipment', 'showUploadSidebar')
    .addToUi();
}

function initializeShipmentSheet() {
  const workbook = SpreadsheetApp.getActiveSpreadsheet();
  const lock = LockService.getDocumentLock();
  lock.waitLock(10000);
  try {
    shipmentSheet_(workbook);
    SpreadsheetApp.flush();
  } finally {
    lock.releaseLock();
  }
}

function showUploadSidebar() {
  initializeShipmentSheet();
  SpreadsheetApp.getUi().showSidebar(
    HtmlService.createHtmlOutputFromFile('Sidebar').setTitle('Upload Shipment')
  );
}

function shipmentSheet_(workbook) {
  let sheet = workbook.getSheetByName(INTAKE_SHEET);
  if (!sheet) {
    const tabs = workbook.getSheets();
    sheet = tabs.length === 1 && tabs[0].getLastRow() === 0
      ? tabs[0].setName(INTAKE_SHEET)
      : workbook.insertSheet(INTAKE_SHEET);
  }
  if (sheet.getLastRow() === 0) {
    sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]).setFontWeight('bold');
    sheet.setFrozenRows(1);
  }
  if (sheet.getRange('A1:B1').getDisplayValues()[0].join('|') !== HEADERS.slice(0, 2).join('|')) {
    throw new Error('The Shipments tab must begin with Origin ZIP and TFWW Account.');
  }
  if (!sheet.getRange('F1').getValue()) {
    sheet.getRange(1, 6, 1, 3).setValues([HEADERS.slice(5)]).setFontWeight('bold');
    sheet.getRange('F2:F').setWrapStrategy(SpreadsheetApp.WrapStrategy.CLIP);
  }
  sheet.getRange('A2:A').setNumberFormat('@');
  sheet.getRange('B2:B').setDataValidation(
    SpreadsheetApp.newDataValidation().requireValueInList(TFWW_ACCOUNTS, true)
      .setAllowInvalid(false).build()
  );
  return sheet;
}

function uploadShipment(form) {
  const workbook = SpreadsheetApp.getActiveSpreadsheet();
  const file = form.document;
  if (!file || !/\.(eml|pdf|png|jpe?g)$/i.test(file.getName() || '')) {
    throw new Error('Choose an EML, PDF, PNG, JPG, or JPEG file.');
  }
  const size = file.getBytes().length;
  if (size === 0 || size > 20 * 1024 * 1024) {
    throw new Error('Choose a nonempty file up to 20 MiB.');
  }

  const response = sendToIntake_(file);
  if (response.getResponseCode() !== 200) {
    throw new Error(intakeError_(response.getResponseCode()));
  }
  const shipment = JSON.parse(response.getContentText());
  const {source, errors, status, ...review} = shipment;
  const reviewJson = JSON.stringify(review);
  if (reviewJson.length > 49000) throw new Error('The shipment has too many details for one sheet row.');
  const zip = shipment.shipper.postal_code || '';
  const notes = shipment.errors.concat(shipment.warnings).join('\n');
  const lock = LockService.getDocumentLock();
  lock.waitLock(10000);
  try {
    const sheet = shipmentSheet_(workbook);
    const row = sheet.getLastRow() + 1;
    if (row > sheet.getMaxRows()) {
      sheet.insertRowsAfter(sheet.getMaxRows(), 1);
      shipmentSheet_(workbook);
    }
    sheet.getRange(row, 1, 1, HEADERS.length).setValues([[
      sheetText_(zip), '', sheetText_(file.getName()), status, sheetText_(notes), reviewJson, '', ''
    ]]);
    sheet.getRange(row, 6).setNote(JSON.stringify({
      expected_total_quantity: review.expected_total_quantity,
      expected_total_quantity_uom: review.expected_total_quantity_uom,
      expected_total_weight: review.expected_total_weight,
      expected_total_weight_uom: review.expected_total_weight_uom
    }));
    SpreadsheetApp.flush();
    return {row: row, hasZip: Boolean(zip)};
  } finally {
    lock.releaseLock();
  }
}

function sheetText_(value) {
  return value ? "'" + value : '';
}

function checkUploadConnection() {
  const probe = Utilities.newBlob('Connection check', 'application/pdf', 'probe.pdf');
  const response = sendToIntake_(probe);
  const code = response.getResponseCode();
  if (code === 422) {
    const data = JSON.parse(response.getContentText());
    if (data.detail === 'probe.pdf does not match its declared file type.') {
      return 'Connected to Cloud Run. Choose a shipment document to upload.';
    }
  }
  throw new Error(intakeError_(code));
}

function sendToIntake_(file) {
  return callIntake_('/extract', {payload: {file: file}});
}

function callIntake_(path, options) {
  const config = PropertiesService.getScriptProperties();
  const baseUrl = (config.getProperty('SHIPMENT_INTAKE_URL') || '').trim().replace(/\/+$/, '');
  const token = config.getProperty('SHIPMENT_INTAKE_TOKEN') ||
    config.getProperty('shipment-intake-token');
  if (!/^https:\/\/[^/?#]+$/.test(baseUrl)) {
    throw new Error('Set SHIPMENT_INTAKE_URL to the Cloud Run service URL in Script properties.');
  }
  if (!token) {
    throw new Error('Set SHIPMENT_INTAKE_TOKEN in Script properties.');
  }
  return UrlFetchApp.fetch(baseUrl + '/shipment-intake' + path, {
    method: 'post',
    headers: {'X-Shipment-Intake-Token': token},
    ...options,
    muteHttpExceptions: true,
    followRedirects: false
  });
}

function intakeError_(code) {
  if (code === 401) return 'HTTP 401. The Script property token and Cloud Run secret must match.';
  if (code === 503) return 'HTTP 503. Cloud Run has not loaded the upload token.';
  if (code === 422) return 'HTTP 422. Cloud Run could not read the document.';
  return 'The upload service returned HTTP ' + code + '.';
}

function selectedShipmentForSubmit() {
  const workbook = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = workbook.getActiveSheet();
  const range = sheet.getActiveRange();
  if (sheet.getName() !== INTAKE_SHEET || !range || range.getNumRows() !== 1 || range.getRow() < 2) {
    throw new Error('Select one shipment row on the Shipments tab.');
  }
  const row = range.getRow();
  const [account, filename] = sheet.getRange(row, 2, 1, 2).getDisplayValues()[0];
  if (!TFWW_ACCOUNTS.includes(account)) throw new Error('Choose an account in column B.');
  return {row: row, account: account, filename: filename};
}

function submitSelectedShipment(selection) {
  const current = selectedShipmentForSubmit();
  if (current.row !== selection.row || current.account !== selection.account ||
      current.filename !== selection.filename) {
    throw new Error('The selection changed. Review and confirm the shipment again.');
  }
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(INTAKE_SHEET);
  const row = current.row;
  const lock = LockService.getDocumentLock();
  let submission;
  let attempt;
  lock.waitLock(10000);
  try {
    const [zip, account, filename, status, , review, , previous] =
      sheet.getRange(row, 1, 1, HEADERS.length).getDisplayValues()[0];
    if (previous || ['SUBMITTING', 'SUBMITTED', 'CHECK_TFWW'].includes(status)) {
      throw new Error('This row already has a submission attempt. Check the TFWW result first.');
    }
    if (account !== selection.account || filename !== selection.filename) {
      throw new Error('The shipment changed. Review and confirm it again.');
    }
    if (!review) throw new Error('This row has no reviewed shipment details. Upload the source again.');
    let shipment;
    try {
      shipment = JSON.parse(review);
      shipment.shipper.postal_code = zip || null;
      const sourceTotals = sheet.getRange(row, 6).getNote();
      if (!sourceTotals) throw new Error('Missing original source totals.');
      Object.assign(shipment, JSON.parse(sourceTotals));
    } catch (error) {
      throw new Error('Column F must contain valid shipment JSON and its original upload note.');
    }
    submission = JSON.stringify({account: account, shipment: shipment});
    attempt = Utilities.getUuid();
    sheet.getRange(row, 4).setValue('SUBMITTING');
    sheet.getRange(row, 7, 1, 2).setValues([['', attempt]]);
    SpreadsheetApp.flush();
  } finally {
    lock.releaseLock();
  }

  let status = 'CHECK_TFWW';
  let result;
  let allowCorrection = false;
  try {
    const response = callIntake_('/submit', {contentType: 'application/json', payload: submission});
    const code = response.getResponseCode();
    const data = JSON.parse(response.getContentText());
    if (code === 200 && Number.isInteger(data.tfww_status)) {
      result = 'TFWW HTTP ' + data.tfww_status + '\n' + data.response;
      if (data.tfww_status >= 200 && data.tfww_status < 300) status = 'SUBMITTED';
    } else {
      result = 'HTTP ' + code + ': ' + JSON.stringify(data.detail || data);
      allowCorrection = [401, 404, 422, 503].includes(code);
      if (allowCorrection) status = code === 422 ? 'VALIDATION_FAILED' : 'API_ERROR';
    }
  } catch (error) {
    result = 'Could not confirm the result. Check TFWW before retrying.';
  }
  lock.waitLock(10000);
  try {
    const marker = sheet.getRange('H2:H').createTextFinder(attempt).matchEntireCell(true).findNext();
    if (!marker) throw new Error('The shipment row moved or was deleted. Check TFWW for the result.');
    const updatedRow = marker.getRow();
    sheet.getRange(updatedRow, 4).setValue(status);
    sheet.getRange(updatedRow, 7).setValue(sheetText_(result.slice(0, 16000)));
    if (allowCorrection) marker.clearContent();
    SpreadsheetApp.flush();
    return {status: status, row: updatedRow};
  } finally {
    lock.releaseLock();
  }
}

function loadShipmentForReview() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getActiveRange();
  if (sheet.getName() !== INTAKE_SHEET || !range || range.getNumRows() !== 1 || range.getRow() < 2) {
    throw new Error('Select one shipment row on the Shipments tab.');
  }
  const row = range.getRow();
  const [zip, , filename, status, notes, review, , attempt] =
    sheet.getRange(row, 1, 1, HEADERS.length).getDisplayValues()[0];
  if (attempt || ['SUBMITTING', 'SUBMITTED', 'CHECK_TFWW'].includes(status)) {
    throw new Error('This row has a submission attempt. Check the TFWW result.');
  }
  if (!review) throw new Error('This row has no shipment details. Upload the source document again.');
  const shipment = JSON.parse(review);
  shipment.shipper.postal_code = zip || null;
  return {row, filename, notes, shipment, original: review, originalZip: zip};
}

function saveShipmentReview(review) {
  const lock = LockService.getDocumentLock();
  lock.waitLock(10000);
  try {
    const current = loadShipmentForReview();
    if (current.row !== review.row || current.filename !== review.filename ||
        current.original !== review.original || current.originalZip !== review.originalZip) {
      throw new Error('The selected row or shipment changed. Open it again before saving.');
    }
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    for (const item of review.shipment.items) {
      item.dimension_uom = item.dimension_uom ?? null;
      item.hazardous_material = item.hazardous_material ?? null;
    }
    const json = JSON.stringify(review.shipment);
    if (json.length > 49000) throw new Error('The shipment has too many details for one sheet row.');
    sheet.getRange(review.row, 1).setValue(sheetText_(review.shipment.shipper.postal_code));
    sheet.getRange(review.row, 4).setValue('REVIEW_REQUIRED');
    sheet.getRange(review.row, 6).setValue(json);
    SpreadsheetApp.flush();
    return review.row;
  } finally {
    lock.releaseLock();
  }
}
