/// <reference path="logger.ts" />

module lib {
    export module utils {
        export module detail {
            import Logger = lib.utils.detail.Logger;

            var logger = new Logger();

            export function assert(res, msg?) {
              try {
                if (!res) {
                    //debugger;
                    if (msg) {
                        logger.error('FAIL: ' + msg);
                    } else {
                        logger.error('FAIL: ');
                    }
                }
              } catch (e) {}
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
        export module assert {
            import _assert = lib.utils.detail.assert;
            export function ok(cond, msg?) {
                return _assert(cond, msg);
            }

            export function fail(cond, msg?) {
                return _assert(!cond, msg);
            }

            export function strictEqual(a, b, msg?) {
                return _assert(a === b, msg);
            }

            export function notStrictEqual(a, b, msg?) {
                return fail(a === b, msg);
            }

            export function deepEqual(a, b, msg?) {
                ok(a === b, msg);

                var aprops = Object.getOwnPropertyNames(a);
                var bprops = Object.getOwnPropertyNames(b);

                ok(aprops === bprops, msg);

                aprops.forEach(function(prop) {
                    ok(a.hasOwnProperty(prop), msg);
                    ok(b.hasOwnProperty(prop), msg);
                    deepEqual(a[prop], b[prop], msg);
                });
            }
        }
    }
}
