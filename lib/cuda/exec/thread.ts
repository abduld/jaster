
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Thread {
        error:utils.Error = new utils.Error();
        threadIdx:utils.Dim3 = new utils.Dim3(0);
        blockIdx:utils.Dim3 = new utils.Dim3(0);
        blockDim:utils.Dim3 = new utils.Dim3(0);
        gridIdx:utils.Dim3 = new utils.Dim3(0);
        gridDim:utils.Dim3 = new utils.Dim3(0);
        block:Block;
        warp:Warp;
        status:cuda.Status;
        fun:Function = undefined;
        args:Array<any> = [];

        constructor(block:Block, threadIdx:utils.Dim3, fun:Function, args:Array<any>) {
            this.status = cuda.Status.Idle;
            this.block = block;
            this.blockIdx = block.blockIdx;
            this.gridIdx = block.gridIdx;
            this.gridDim = block.gridDim;
            this.threadIdx = threadIdx;
            this.args = args;
            this.fun = fun;
        }

        run() {
            var res:cuda.Status;
            this.status = cuda.Status.Running;
            try {
                res = this.fun.apply(this, this.args);
            } catch (err) {
                res = err.code;
            }
            this.status = cuda.Status.Complete;
            return res;
        }

        terminate() {
            this.status = cuda.Status.Stopped;
        }
    }
}