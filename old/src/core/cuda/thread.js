/// <reference path="../../ref.ts" />
define(["require", "exports", "./cuda", "./../../utils/utils"], function(require, exports, cuda, utils) {
    var Thread = (function () {
        function Thread(block, threadIdx, fun, args) {
            this.error = new utils.Error();
            this.threadIdx = new utils.Dim3(0);
            this.blockIdx = new utils.Dim3(0);
            this.blockDim = new utils.Dim3(0);
            this.gridIdx = new utils.Dim3(0);
            this.gridDim = new utils.Dim3(0);
            this.fun = undefined;
            this.args = [];
            this.status = 1 /* Idle */;
            this.block = block;
            this.blockIdx = block.blockIdx;
            this.gridIdx = block.gridIdx;
            this.gridDim = block.gridDim;
            this.threadIdx = threadIdx;
            this.args = args;
            this.fun = fun;
        }
        Thread.prototype.run = function () {
            this.status = 0 /* Running */;
            var res = this.fun.apply(this, this.args);
            this.status = 2 /* Complete */;
            return res;
        };
        return Thread;
    })();

    
    return Thread;
});
//# sourceMappingURL=thread.js.map
