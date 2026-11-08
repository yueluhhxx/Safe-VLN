"""Pure checkpoint publication gates; no rollout or optimizer scheduling.

Bounds are inclusive. Zero ``end`` means unbounded; zero ``interval`` keeps
every existing save call within the window. A positive interval is anchored
at ``start``. These helpers never create extra save calls or delete artifacts.
"""

import operator


def _nonnegative_integer(value, name):
    # bool is an int subclass, but accepting True as a cadence is a typo trap.
    if isinstance(value, bool):
        raise ValueError('{} must be a nonnegative integer, not bool'.format(name))
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError('{} must be a nonnegative integer'.format(name)) from error
    if result < 0:
        raise ValueError('{} must be nonnegative'.format(name))
    return result


def validate_checkpoint_cadence(start=0, end=0, interval=0):
    """Validate configuration and return its normalized integer triple."""

    start = _nonnegative_integer(start, 'checkpoint_start_iter')
    end = _nonnegative_integer(end, 'checkpoint_end_iter')
    interval = _nonnegative_integer(interval, 'checkpoint_interval')
    if end != 0 and end < start:
        raise ValueError('checkpoint_end_iter must be zero or >= checkpoint_start_iter')
    return start, end, interval


def validate_checkpoint_save_calls(start, end, interval, save_call_interval):
    """Reject a fresh-run logging cadence that would silently miss saves.

    The existing trainer calls save_checkpoint once per log interval. A gate
    cannot create the missing calls, so both the start anchor and requested
    positive checkpoint interval must be reachable by that existing schedule.
    Legacy interval=0 is left unchanged, including arbitrary window bounds.
    """

    start, end, interval = validate_checkpoint_cadence(start, end, interval)
    if interval == 0:
        return
    save_call_interval = _nonnegative_integer(save_call_interval, 'IL.log_every')
    if save_call_interval == 0:
        raise ValueError('IL.log_every must be positive for an explicit checkpoint cadence')
    if start % save_call_interval or interval % save_call_interval:
        raise ValueError('IL.log_every would skip requested checkpoint iterations')


def should_save_checkpoint(iteration, start=0, end=0, interval=0):
    """Decide whether an existing save request belongs to the configured set."""

    start, end, interval = validate_checkpoint_cadence(start, end, interval)
    iteration = _nonnegative_integer(iteration, 'checkpoint iteration')
    if iteration < start or (end != 0 and iteration > end):
        return False
    return interval == 0 or (iteration - start) % interval == 0
