
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Block {
        grid:Grid;
        blockIdx:utils.Dim3 = new utils.Dim3(0);
        blockDim:utils.Dim3 = new utils.Dim3(0);
        gridIdx:utils.Dim3 = new utils.Dim3(0);
        gridDim:utils.Dim3 = new utils.Dim3(0);
        threads:Thread[] = null;
        barriers:Barrier[] = null;
        status:cuda.Status;
        fun:Function = undefined;
        args:Array<any> = [];

        constructor(grid:Grid, blockIdx:utils.Dim3, fun:Function, args:Array<any>) {
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
