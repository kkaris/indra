import sys
import logging


codec_options = ['utf-8', 'latin_1']


logger = logging.getLogger(__name__)


def try_decode(byte_string, codec):
    try:
        s = byte_string.decode(codec)
        return s
    except:
        return None


def shortest_string(strings):
    best_string = None
    best_length = None

    for s in strings:
        if best_string is None or len(s) < best_length:
            best_string = s
            best_length = len(s)
    return best_string


def fix_character_encoding(input_file, output_file):
    with open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            for line in f_in:
                # Try to decode with both latin_1 and utf-8
                decoded = [try_decode(line, c) for c in codec_options]

                decoded = [d for d in decoded if d is not None]

                if len(decoded) == 0:
                    # Hopefully at least one codec worked
                    logger.info('Could not decode: %s' % line)
                    sys.exit(1)
                else:
                    # If more than one, choose the codec that gives the best
                    # length
                    chosen_string = shortest_string(decoded)

                    # Write result as ascii, with non-ascii characters escaped
                    f_out.write(chosen_string.encode('utf-8'))


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        logger.error('Expected two arguments: the input file'
                     ' and the output file')
        sys.exit(1)
    input_file = args[0]
    output_file = args[1]

    fix_character_encoding(input_file, output_file)

