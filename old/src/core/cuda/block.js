/// <reference path="../../ref.ts" />
define(["require", "exports", "./../../utils/utils"], function(require, exports, utils) {
    var Block = (function () {
        function Block() {
            this.blockIdx = new utils.Dim3(0);
            this.blockDim = new utils.Dim3(0);
            this.gridIdx = new utils.Dim3(0);
            this.gridDim = new utils.Dim3(0);
            this.threads = null;
        }
        return Block;
    })();

    
    return Block;
});
//# sourceMappingURL=block.js.map
