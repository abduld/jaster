/*
 Copyright (c) 2014 Sebastian McKenzie

 MIT License

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

module lib {
    export module  utils {
        function trimRight(str) {
            return str.replace(/[\n\s]+$/g, "");
        }

        function repeat(width, cha) {
            cha = cha || " ";
            return new Array(width + 1).join(cha);
        }

        export class Buffer {
            buf:string;
            format:any;
            position:any;
            private _indent:any;

            constructor(s, format?) {
                this.position = s;
                if (_.isUndefined(format)) {
                    format = {
                        parentheses: true,
                        semicolons: true,
                        comments: true,
                        compact: false,
                        indent: {
                            adjustMultilineComment: true,
                            style: "  ",
                            base: 0
                        }
                    };
                }

                this._indent = format.indent.base;
                this.format = format;
                this.buf = s;
            }

            get() {
                return trimRight(this.buf);
            }

            getIndent() {
                if (this.format.compact) {
                    return "";
                } else {
                    return repeat(this._indent, this.format.indent.style);
                }
            }

            indentSize() {
                return this.getIndent().length;
            }

            indent() {
                this._indent++;
            }

            dedent() {
                this._indent--;
            }

            semicolon() {
                if (this.format.semicolons) this.push(";");
            }

            ensureSemicolon() {
                if (!this.isLast(";")) this.semicolon();
            }

            rightBrace() {
                this.newline(true);
                this.push("}");
            }

            keyword(name) {
                this.push(name);
                this.push(" ");
            }

            space() {
                if (this.buf && !this.isLast([" ", "\n"])) {
                    this.push(" ");
                }
            }

            removeLast(cha) {
                if (!this.isLast(cha)) return;

                this.buf = this.buf.slice(0, -1);
                this.position.unshift(cha);
            }

            newline(i, removeLast?) {
                if (!this.buf) return;
                if (this.format.compact) return;
                if (this.endsWith("{\n")) return;

                if (_.isBoolean(i)) {
                    removeLast = i;
                    i = null;
                }

                if (_.isNumber(i)) {
                    if (this.endsWith(repeat(i, "\n"))) return;

                    var self = this;
                    _.times(i, function () {
                        self.newline(null, removeLast);
                    });
                    return;
                }

                if (removeLast && this.isLast("\n")) this.removeLast("\n");

                this.removeLast(" ");
                this.buf = this.buf.replace(/\n +$/, "\n");
                this._push("\n");
            }

            push(str, noIndent?) {
                if (this._indent && !noIndent && str !== "\n") {
                    // we have an indent level and we aren't pushing a newline
                    var indent = this.getIndent();

                    // replace all newlines with newlines with the indentation
                    str = str.replace(/\n/g, "\n" + indent);

                    // we've got a newline before us so prepend on the indentation
                    if (this.isLast("\n")) str = indent + str;
                }

                this._push(str);
            }

            _push(str) {
                this.position.push(str);
                this.buf += str;
            }

            endsWith(str) {
                return this.buf.slice(-str.length) === str;
            }

            isLast(cha, trimRight?) {
                var buf = this.buf;
                if (trimRight) buf = trimRight(buf);

                var chars = [].concat(cha);
                return _.contains(chars, _.last(buf));
            }

            toString(typ ? : string) : string{
                if (_.isUndefined(typ)) {
                    typ = "ASCII";
                }
                typ = typ.toUpperCase();
                switch(typ) {
                    case "ASCII":
                        return this.buf;
                    case "BASE64":
                        return lib.utils.base64Encode(this.buf);
                }
            }

        }
    }
}