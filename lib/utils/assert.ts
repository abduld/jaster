
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

            export function assertUnreachable(msg: string): void {
                var location = new Error().stack.split('\n')[1];
                throw new Error("Reached unreachable location " + location + msg);
            }

            export function error(message: string) {
                console.error(message);
                throw new Error(message);
            }
            export function assertNotImplemented(condition: boolean, message: string) {
                if (!condition) {
                    error("notImplemented: " + message);
                }
            }
        }
        export import assert = detail.assert;
    }
}