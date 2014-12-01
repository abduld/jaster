var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            (function (LogType) {
                LogType[LogType["Debug"] = 0] = "Debug";
                LogType[LogType["Trace"] = 1] = "Trace";
                LogType[LogType["Warn"] = 2] = "Warn";
                LogType[LogType["Error"] = 3] = "Error";
                LogType[LogType["Fatal"] = 4] = "Fatal";
            })(detail.LogType || (detail.LogType = {}));
            var LogType = detail.LogType;
            var Logger = (function () {
                function Logger(level) {
                    if (level) {
                        this._level = level;
                    }
                    else {
                        this._level = 0 /* Debug */;
                    }
                }
                Logger.prototype._go = function (msg, type) {
                    var color = {
                        "LogType.Debug": '\033[39m',
                        "LogType.Trace": '\033[39m',
                        "LogType.Warn": '\033[33m',
                        "LogType.Error": '\033[33m',
                        "LogType.Fatal": '\033[31m'
                    };
                    if (type >= this._level) {
                        switch (type) {
                            case 0 /* Debug */:
                            case 1 /* Trace */:
                                console.info(msg);
                                break;
                            case 2 /* Warn */:
                                console.warn(msg);
                                break;
                            case 3 /* Error */:
                            case 4 /* Fatal */:
                            default:
                                debugger;
                                console.error(msg);
                                break;
                        }
                    }
                };
                Logger.prototype.debug = function (msg) {
                    this._go(msg, 0 /* Debug */);
                };
                Logger.prototype.trace = function (msg) {
                    this._go(msg, 1 /* Trace */);
                };
                Logger.prototype.warn = function (msg) {
                    this._go(msg, 2 /* Warn */);
                };
                Logger.prototype.error = function (msg) {
                    this._go(msg, 3 /* Error */);
                };
                Logger.prototype.fatal = function (msg) {
                    this._go(msg, 4 /* Fatal */);
                };
                return Logger;
            })();
            detail.Logger = Logger;
        })(detail = utils.detail || (utils.detail = {}));
        utils.logger = new detail.Logger();
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
