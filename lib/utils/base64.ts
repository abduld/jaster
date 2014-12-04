module lib.utils {
    export module base64 {

        var charToIntMap = {};
        var intToCharMap = {};

        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
            .split('')
            .forEach(function (ch, index) {
                charToIntMap[ch] = index;
                intToCharMap[index] = ch;
            });

        /**
         * Encode an integer in the range of 0 to 63 to a single base 64 digit.
         */
        export function encode(aNumber) {
            if (aNumber in intToCharMap) {
                return intToCharMap[aNumber];
            }
            throw new TypeError("Must be between 0 and 63: " + aNumber);
        };

        /**
         * Decode a single base 64 digit to an integer.
         */
        export function decode(aChar) {
            if (aChar in charToIntMap) {
                return charToIntMap[aChar];
            }
            throw new TypeError("Not a valid base 64 digit: " + aChar);
        };
    }
}