/// <reference path="../ref.ts" />

module lib {
    export module parallel {
        export class Thread {
            private id: string;

            private threadIdx: lib.cuda.Dim3;
            private blockIdx: lib.cuda.Dim3;
            private blockDim: lib.cuda.Dim3;
            private gridDim: lib.cuda.Dim3;
            private gridIdx: lib.cuda.Dim3;

            constructor(id: string) {
                this.id = id;
            }
        }
    }
}