
/// <reference path="logger.ts" />

module lib {
    export module utils {
        export module detail {
            import Logger = lib.utils.detail.Logger;

            var logger = new Logger();

            export function assert(res, msg) {
                if (!res) {
                    logger.error('FAIL: ' + msg);
                } else {
                    logger.debug('Pass: ' + msg);
                }
            }
        }
    }
}