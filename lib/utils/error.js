var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            (function (ErrorCode) {
                ErrorCode[ErrorCode["Success"] = 0] = "Success";
                ErrorCode[ErrorCode["MemoryOverflow"] = 1] = "MemoryOverflow";
                ErrorCode[ErrorCode["IntegerOverflow"] = 2] = "IntegerOverflow";
                ErrorCode[ErrorCode["Unknown"] = 3] = "Unknown";
                ErrorCode[ErrorCode["Message"] = 4] = "Message";
            })(detail.ErrorCode || (detail.ErrorCode = {}));
            var ErrorCode = detail.ErrorCode;
        })(detail = utils.detail || (utils.detail = {}));
        ;
        var Error = (function () {
            function Error(arg) {
                if (arg) {
                    if (utils.isString(arg)) {
                        this.message = arg;
                        this.code = 4 /* Message */;
                    }
                    else {
                        this.code = arg;
                        this.message = arg.toString();
                    }
                }
                else {
                    this.code = 0 /* Success */;
                    this.message = "Success";
                }
            }

            return Error;
        })();
        utils.Error = Error;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
//# sourceMappingURL=error.js.map