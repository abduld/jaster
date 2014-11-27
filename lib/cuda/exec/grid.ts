
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Grid {
        public gridIdx:utils.Dim3 = new utils.Dim3(0);
        public gridDim:utils.Dim3 = new utils.Dim3(0);
        public blocks:Block[] = null;
    }
}