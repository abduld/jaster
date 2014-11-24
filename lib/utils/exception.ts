
/// <reference path="../ref.ts" />


module Core {
    class Exception {
        private error_ : any;
        constructor(msg : string, name : string = 'Exception') {
            this.error_ = new Error(msg);
            this.error_.name = name;
        }
        message() : string {
            return this.error_.message;
        }
        name() : string {
            return this.error_.name;
        }
        stackTrace() : string  {
            return this.error_.stack;
        }
        toString() : string {
            return this.error_.name + ': ' + this.error_.message;
        }
    }
}

export = Core;