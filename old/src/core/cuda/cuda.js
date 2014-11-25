/// <reference path="block.ts" />
/// <reference path="thread.ts" />
/// <reference path="grid.ts" />
/// <reference path="warp.ts" />
define(["require", "exports"], function(require, exports) {
    (function (Status) {
        Status[Status["Running"] = 0] = "Running";
        Status[Status["Idle"] = 1] = "Idle";
        Status[Status["Complete"] = 2] = "Complete";
        Status[Status["Stopped"] = 3] = "Stopped";
    })(exports.Status || (exports.Status = {}));
    var Status = exports.Status;
    ;
});
//# sourceMappingURL=cuda.js.map
