/// <reference path="../../ref.ts" />
define(["require", "exports", "./../../utils/utils"], function(require, exports, utils) {
    var Warp = (function () {
        function Warp() {
            this.id = utils.guuid();
            this.thread = null;
        }
        return Warp;
    })();

    
    return Warp;
});
//# sourceMappingURL=warp.js.map
