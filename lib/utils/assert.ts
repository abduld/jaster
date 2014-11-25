
/// <reference path="logger.ts" />

import Logger = require("./logger");

var logger = new Logger();

function assert(res, msg) {
    if (!res) {
        logger.error('FAIL: ' + msg);
    } else {
        logger.debug('Pass: ' + msg);
    }
}

export = assert;