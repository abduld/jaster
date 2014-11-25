/// <reference path="logger.ts" />
define(["require", "exports", "./logger"], function (require, exports, Logger) {
    var logger = new Logger();
    function assert(res, msg) {
        if (!res) {
            logger.error('FAIL: ' + msg);
        }
        else {
            logger.debug('Pass: ' + msg);
        }
    }
    return assert;
});
