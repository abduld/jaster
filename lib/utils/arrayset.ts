/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */

module lib {
    export module utils {
        module internal {

            /**
             * Because behavior goes wacky when you set `__proto__` on objects, we
             * have to prefix all the strings in our set with an arbitrary character.
             *
             * See https://github.com/mozilla/source-map/pull/31 and
             * https://github.com/mozilla/source-map/issues/30
             *
             * @param String aStr
             */
            export function toSetString(aStr) {
                return '$' + aStr;
            }


            export function fromSetString(aStr) {
                return aStr.substr(1);
            }
        }
        /**
         * A data structure which is a combination of an array and a set. Adding a new
         * member is O(1), testing for membership is O(1), and finding the index of an
         * element is O(1). Removing elements from the set is not supported. Only
         * strings are supported for membership.
         */
        export class ArraySet {
            private _array:string[] = [];
            private _set:{ [key: string]: number; } = {};

            constructor() {
                this._array = [];
                this._set = {};
            }

            /**
             * Static method for creating ArraySet instances from an existing array.
             */
            static fromArray(aArray, aAllowDuplicates:boolean) {
                var set = new ArraySet();
                for (var i = 0, len = aArray.length; i < len; i++) {
                    set.add(aArray[i], aAllowDuplicates);
                }
                return set;
            }

            /**
             * Add the given string to this set.
             *
             * @param String aStr
             */
            add(aStr:string, aAllowDuplicates?:boolean) {
                var isDuplicate = this.has(aStr);
                var idx = this._array.length;
                if (!isDuplicate || aAllowDuplicates) {
                    this._array.push(aStr);
                }
                if (!isDuplicate) {
                    this._set[internal.toSetString(aStr)] = idx;
                }
            }

            /**
             * Is the given string a member of this set?
             *
             * @param String aStr
             */
            has(aStr:string):boolean {
                return Object.prototype.hasOwnProperty.call(this._set,
                    internal.toSetString(aStr));
            }

            /**
             * What is the index of the given string in the array?
             *
             * @param String aStr
             */
            indexOf(aStr:string) {
                if (this.has(aStr)) {
                    return this._set[internal.toSetString(aStr)];
                }
                throw new Error('"' + aStr + '" is not in the set.');
            }

            /**
             * What is the element at the given index?
             *
             * @param Number aIdx
             */
            at(aIdx:number):string {
                if (aIdx >= 0 && aIdx < this._array.length) {
                    return this._array[aIdx];
                }
                throw new Error('No element indexed by ' + aIdx);
            }

            /**
             * Returns the array representation of this set (which has the proper indices
             * indicated by indexOf). Note that this is a copy of the internal array used
             * for storing the members so that no one can mess with internal state.
             */
            toArray():string[] {
                return this._array.slice();
            }
        }
    }
}