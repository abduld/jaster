
/// <reference path="../ref.ts" />

import log = require("./log");

var logger = new log();

function assert(res, msg) {
  if (!res) {
    error('FAIL: ' + msg);
  } else {
    log('Pass: ' + msg);
  }
}

export = assert;