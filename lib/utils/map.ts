/*
 Copyright (C) 2012 Yusuke Suzuki <utatane.tea@gmail.com>
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*global module:true*/
module lib.utils {
    export class Map<K, V> {
        private __data;

        constructor() {
            this.__data = {};
        }

        get(key: K): V {
            var skey = '$' + key;
            if (this.__data.hasOwnProperty(skey)) {
                return this.__data[skey];
            }
        }

        has(key: K) {
            var skey = '$' + key;
            return this.__data.hasOwnProperty(skey);
        }

        set(key: K, val: V) {
            var skey = '$' + key;
            this.__data[skey] = val;
        }

        delete(key: K) {
            var skey = '$' + key;
            return delete this.__data[skey];
        }

        clear() {
            this.__data = {};
        }

        forEach(callback, thisArg) {
            var real, key: K;
            for (real in this.__data) {
                if (this.__data.hasOwnProperty(real)) {
                    key = real.substring(1);
                    callback.call(thisArg, this.__data[real], key, this);
                }
            }
        }

        keys() {
            var real, result;
            result = [];
            for (real in this.__data) {
                if (this.__data.hasOwnProperty(real)) {
                    result.push(real.substring(1));
                }
            }
            return result;
        }

        values() {
            var real, result;
            result = [];
            for (real in this.__data) {
                if (this.__data.hasOwnProperty(real)) {
                    result.push(this.__data[real]);
                }
            }
            return result;
        }

        items() {
            var real, result;
            result = [];
            for (real in this.__data) {
                if (this.__data.hasOwnProperty(real)) {
                    result.push([real.substring(1), this.__data[real]]);
                }
            }
            return result;
        }
    }

}