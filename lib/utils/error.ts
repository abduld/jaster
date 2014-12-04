module lib.utils {
    export module detail {
        export enum ErrorCode {
            Success,
            MemoryOverflow,
            IntegerOverflow,
            Unknown,
            Message
        }
    }
    ;
    export class Error {
        code:detail.ErrorCode;
        message:string;
        stack:any;

        constructor(code?:detail.ErrorCode);
        constructor(str?:string);
        constructor(arg?:any) {
            if (arg) {
                if (isString(arg)) {
                    this.message = arg;
                    this.code = detail.ErrorCode.Message;
                } else {
                    this.code = arg;
                    this.message = arg.toString();
                }
            } else {
                this.code = detail.ErrorCode.Success;
                this.message = "Success";
            }
        }
    }
}