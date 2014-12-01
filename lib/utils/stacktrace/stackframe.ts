
/*This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
*/
// from https://github.com/stacktracejs/stackframe
module lib.utils {
    export module stacktrace {
        function _isNumber(n) {
            return !isNaN(parseFloat(n)) && isFinite(n);
        }

        export class StackFrame {
            fileName: string;
            functionName: string;
            args: any[];
            lineNumber: number;
            columnNumber: number;
            constructor(functionName, args, fileName, lineNumber, columnNumber) {
                if (functionName !== undefined) {
                    this.setFunctionName(functionName);
                }
                if (args !== undefined) {
                    this.setArgs(args);
                }
                if (fileName !== undefined) {
                    this.setFileName(fileName);
                }
                if (lineNumber !== undefined) {
                    this.setLineNumber(lineNumber);
                }
                if (columnNumber !== undefined) {
                    this.setColumnNumber(columnNumber);
                }
            }
            getFunctionName() {
                return this.functionName;
            }
            setFunctionName(v: string) {
                this.functionName = String(v);
            }

            getArgs() {
                return this.args;
            }
            setArgs(v: any[]) {
                if (Object.prototype.toString.call(v) !== '[object Array]') {
                    throw new TypeError('Args must be an Array');
                }
                this.args = v;
            }

            // NOTE: Property name may be misleading as it includes the path,
            // but it somewhat mirrors V8's JavaScriptStackTraceApi
            // https://code.google.com/p/v8/wiki/JavaScriptStackTraceApi
            getFileName() {
                return this.fileName;
            }
            setFileName(v: string) {
                this.fileName = String(v);
            }

            getLineNumber() {
                return this.lineNumber;
            }
            setLineNumber(v: number) {
                if (!_isNumber(v)) {
                    throw new TypeError('Line Number must be a Number');
                }
                this.lineNumber = Number(v);
            }

            getColumnNumber() {
                return this.columnNumber;
            }
            setColumnNumber(v: number) {
                if (!_isNumber(v)) {
                    throw new TypeError('Column Number must be a Number');
                }
                this.columnNumber = Number(v);
            }

            toString() {
                var functionName = this.getFunctionName() || '{anonymous}';
                var args = '(' + (this.getArgs() || []).join(',') + ')';
                var fileName = this.getFileName() ? ('@' + this.getFileName()) : '';
                var lineNumber = _isNumber(this.getLineNumber()) ? (':' + this.getLineNumber()) : '';
                var columnNumber = _isNumber(this.getColumnNumber()) ? (':' + this.getColumnNumber()) : '';
                return functionName + args + fileName + lineNumber + columnNumber;
            }
        }

    }
}