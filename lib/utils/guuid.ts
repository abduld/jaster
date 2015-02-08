/// <reference path="../ref.ts" />
/// based on https://github.com/broofa/node-uuid/blob/master/uuid.js

module lib {
    export module utils {
        export module detail {
            var randArray = new Uint8Array(16);
            var makeRandom = function() {
                for (var i = 0, r; i < 16; i++) {
                    if ((i & 0x03) === 0) r = Math.random() * 0x100000000;
                    randArray[i] = r >>> ((i & 0x03) << 3) & 0xff;
                }

                return randArray;
            };
            // Maps for number <-> hex string conversion
            var byteToHex: string[] = [];
            var hexToByte: { [key: string]: number; } = {};
            for (var i = 0; i < 256; i++) {
                byteToHex[i] = (i + 0x100).toString(16).substr(1);
                hexToByte[byteToHex[i]] = i;
            }
            // **`unparse()` - Convert UUID byte array (ala parse()) into a string*
            function unparse(buf): string {
                var i = 0, bth = byteToHex;
                return bth[buf[i++]] + bth[buf[i++]] +
                    bth[buf[i++]] + bth[buf[i++]] + '-' +
                    bth[buf[i++]] + bth[buf[i++]] + '-' +
                    bth[buf[i++]] + bth[buf[i++]] + '-' +
                    bth[buf[i++]] + bth[buf[i++]] + '-' +
                    bth[buf[i++]] + bth[buf[i++]] +
                    bth[buf[i++]] + bth[buf[i++]] +
                    bth[buf[i++]] + bth[buf[i++]];
            }

            export function guuid(): string {
                var rnds = makeRandom();
                rnds[6] = (rnds[6] & 0x0f) | 0x40;
                rnds[8] = (rnds[8] & 0x3f) | 0x80;
                return unparse(rnds);
            }
        }
        export import guuid = detail.guuid;
    }
}
