
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Block {
        grid: Grid;
        blockIdx: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        blockDim: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        gridIdx: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        gridDim: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        threads: Thread[] = null;
        barriers: Barrier[] = null;
        status: cuda.Status;
        fun: Function = undefined;
        args: Array<any> = [];

        constructor(grid: Grid, blockIdx: lib.cuda.Dim3, fun: Function, args: Array<any>) {
            this.status = cuda.Status.Idle;
            this.grid = grid;
            this.blockIdx = blockIdx;
            this.gridIdx = grid.gridIdx;
            this.gridDim = grid.gridDim;
            this.args = args;
            this.fun = fun;
        }
    }
}
