/// <reference path="logger.ts" />
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            var Logger = lib.utils.detail.Logger;
            var logger = new Logger();
            function assert(res, msg) {
                if (!res) {
                    debugger;
                    if (msg) {
                        logger.error('FAIL: ' + msg);
                    }
                    else {
                        logger.error('FAIL: ');
                    }
                }
            }
            detail.assert = assert;
            function assertUnreachable(msg) {
                var location = new utils.Error().stack.split('\n')[1];
                throw new utils.Error("Reached unreachable location " + location + msg);
            }
            detail.assertUnreachable = assertUnreachable;
            function error(message) {
                console.error(message);
                throw new utils.Error(message);
            }
            detail.error = error;
            function assertNotImplemented(condition, message) {
                if (!condition) {
                    error("notImplemented: " + message);
                }
            }
            detail.assertNotImplemented = assertNotImplemented;
        })(detail = utils.detail || (utils.detail = {}));
        var assert;
        (function (assert) {
            var _assert = lib.utils.detail.assert;
            function ok(cond, msg) {
                return _assert(cond, msg);
            }
            assert.ok = ok;
            function fail(cond, msg) {
                return _assert(!cond, msg);
            }
            assert.fail = fail;
            function strictEqual(a, b, msg) {
                return _assert(a === b, msg);
            }
            assert.strictEqual = strictEqual;
            function notStrictEqual(a, b, msg) {
                return fail(a === b, msg);
            }
            assert.notStrictEqual = notStrictEqual;
            function deepEqual(a, b, msg) {
                ok(a === b, msg);
                var aprops = Object.getOwnPropertyNames(a);
                var bprops = Object.getOwnPropertyNames(b);
                ok(aprops === bprops, msg);
                aprops.forEach(function (prop) {
                    ok(a.hasOwnProperty(prop), msg);
                    ok(b.hasOwnProperty(prop), msg);
                    deepEqual(a[prop], b[prop], msg);
                });
            }
            assert.deepEqual = deepEqual;
        })(assert = utils.assert || (utils.assert = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
//# sourceMappingURL=assert.js.map