
module lib.utils {
    export module detail {
        export enum ErrorCode {
            Success,
            MemoryOverflow,
            IntegerOverflow,
            Unknown
        }
    };
    export class Error {
        code : detail.ErrorCode;
        constructor(code ? : detail.ErrorCode) {
            if (code) {
                this.code = code;
            } else {
                this.code = detail.ErrorCode.Success;
            }
        }
    }
}