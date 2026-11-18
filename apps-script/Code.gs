const INTAKE_SHEET = 'Shipments';
const TFWW_ACCOUNTS = ['Covenant', 'Ultra Play'];
const HEADERS = ['Origin ZIP', 'TFWW Account', 'Source File', 'Status', 'Review Notes'];

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

  const config = PropertiesService.getScriptProperties();
  const baseUrl = (config.getProperty('SHIPMENT_INTAKE_URL') || '').trim().replace(/\/+$/, '');
  const token = config.getProperty('SHIPMENT_INTAKE_TOKEN');
  if (!/^https:\/\/[^/?#]+$/.test(baseUrl) || !token) {
    throw new Error('The upload service is not connected. Ask the project owner to configure it.');
  }
  const response = UrlFetchApp.fetch(baseUrl + '/shipment-intake/extract', {
    method: 'post',
    headers: {'X-Shipment-Intake-Token': token},
    payload: {file: file},
    muteHttpExceptions: true,
    followRedirects: false
  });
  if (response.getResponseCode() !== 200) {
    throw new Error('The upload service returned HTTP ' + response.getResponseCode() + '.');
  }
  const shipment = JSON.parse(response.getContentText());
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
      sheetText_(zip), '', sheetText_(file.getName()), shipment.status, sheetText_(notes)
    ]]);
    SpreadsheetApp.flush();
    return {row: row, hasZip: Boolean(zip)};
  } finally {
    lock.releaseLock();
  }
}

function sheetText_(value) {
  return value ? "'" + value : '';
}
