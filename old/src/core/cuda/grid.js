/// <reference path="../../ref.ts" />
define(["require", "exports", "./../../utils/utils"], function(require, exports, utils) {
    var Grid = (function () {
        function Grid() {
            this.gridIdx = new utils.Dim3(0);
            this.gridDim = new utils.Dim3(0);
            this.blocks = null;
        }
        return Grid;
    })();

    
    return Grid;
});
//# sourceMappingURL=grid.js.map
