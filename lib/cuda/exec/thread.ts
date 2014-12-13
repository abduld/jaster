/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Thread {
        error: lib.utils.Error = new lib.utils.Error();
        threadIdx: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        blockIdx: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        blockDim: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        gridIdx: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        gridDim: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        block: Block;
        warp: Warp;
        status: cuda.Status;
        fun: Function = undefined;
        args: Array<any> = [];

        constructor(block: Block, threadIdx: lib.cuda.Dim3, fun: Function, args: Array<any>) {
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
            var res: cuda.Status;
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