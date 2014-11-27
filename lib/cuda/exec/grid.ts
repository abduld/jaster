
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export class Grid {
        public gridIdx: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        public gridDim: lib.cuda.Dim3 = new lib.cuda.Dim3(0);
        public blocks:Block[] = null;
    }
}