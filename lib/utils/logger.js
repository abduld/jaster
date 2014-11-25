define(["require", "exports"], function (require, exports) {
    var LogType;
    (function (LogType) {
        LogType[LogType["Debug"] = 0] = "Debug";
        LogType[LogType["Trace"] = 1] = "Trace";
        LogType[LogType["Warn"] = 2] = "Warn";
        LogType[LogType["Error"] = 3] = "Error";
        LogType[LogType["Fatal"] = 4] = "Fatal";
    })(LogType || (LogType = {}));
    var Logger = (function () {
        function Logger(level) {
            if (level) {
                this._level = level;
            }
            else {
                this._level = 0 /* Debug */;
            }
        }
        Logger.prototype.go = function (msg, type) {
            var color = {
                "LogType.Debug": '\033[39m',
                "LogType.Trace": '\033[39m',
                "LogType.Warn": '\033[33m',
                "LogType.Error": '\033[33m',
                "LogType.Fatal": '\033[31m'
            };
            if (type >= this._level) {
                console[type](color[type.toString()] + msg + color["LogType.Debug"]);
            }
        };
        Logger.prototype.debug = function (msg) {
            this.go(msg, 0 /* Debug */);
        };
        Logger.prototype.trace = function (msg) {
            this.go(msg, 1 /* Trace */);
        };
        Logger.prototype.warn = function (msg) {
            this.go(msg, 2 /* Warn */);
        };
        Logger.prototype.error = function (msg) {
            this.go(msg, 3 /* Error */);
        };
        Logger.prototype.fatal = function (msg) {
            this.go(msg, 4 /* Fatal */);
        };
        return Logger;
    })();
    ;
    return Logger;
});
