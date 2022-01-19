def expand_range(s):
  ranges = [span.split('-') for span in s.strip().split(',')]
  ranges = [range(int(span[0]), int(span[1]) + 1) if len(span) == 2 else span for span in ranges]
  return [int(x) for span in ranges for x in span]