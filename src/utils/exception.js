/// <reference path="ref.ts" />
define(["require", "exports"], function (require, exports) {
    var Core;
    (function (Core) {
        var Exception = (function () {
            function Exception(msg, name) {
                if (name === void 0) { name = 'Exception'; }
                this.error_ = new Error(msg);
                this.error_.name = name;
            }
            Exception.prototype.message = function () {
                return this.error_.message;
            };
            Exception.prototype.name = function () {
                return this.error_.name;
            };
            Exception.prototype.stackTrace = function () {
                return this.error_.stack;
            };
            Exception.prototype.toString = function () {
                return this.error_.name + ': ' + this.error_.message;
            };
            return Exception;
        })();
    })(Core || (Core = {}));
    return Core;
});
//# sourceMappingURL=exception.js.map