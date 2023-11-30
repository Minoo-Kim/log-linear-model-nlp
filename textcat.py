#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
import numpy
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "positive_model",
        type=Path,
        help="path to the trained model for positive outcome",
    )
    parser.add_argument(
        "negative_model",
        type=Path,
        help="path to the trained model for negative outcome",
    )
    parser.add_argument(
        "prior_probability",
        type=float,
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    log.info("Testing...")
    pos_lm = LanguageModel.load(args.positive_model)
    neg_lm = LanguageModel.load(args.negative_model)

    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file log-probabilities:")
    gen_counter = 0
    spam_counter = 0
    for file in args.test_files:
        log_prob_pos: float = file_log_prob(file, pos_lm)
        log_prob_neg: float = file_log_prob(file, neg_lm)
        prior_prob_log = math.log(args.prior_probability)
        if (math.exp(log_prob_pos + prior_prob_log - numpy.logaddexp(log_prob_pos + prior_prob_log, log_prob_neg + math.log(1- args.prior_probability))) >= 0.5):
            gen_counter += 1
            print(str(args.positive_model) + "  " + str(file))
        else:
            spam_counter += 1
            print(str(args.negative_model) + "  " + str(file))
    gen_prob = " (" + str(round((gen_counter / (gen_counter + spam_counter) * 100), 2)) + "%" + ")"
    spam_prob = " (" + str(round((spam_counter / (gen_counter + spam_counter) * 100), 2)) + "%" + ")"
    print(str(gen_counter) + " files were probably " + str(args.positive_model) +  gen_prob)
    print(str(spam_counter) + " files were probably " + str(args.negative_model) +  spam_prob)

if __name__ == "__main__":
    main()
