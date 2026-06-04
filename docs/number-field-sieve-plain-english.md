# How This Program Cracks Huge Numbers — in Plain English

This is the no-math, no-jargon version. If you'd like the real equations
afterward, the rigorous companion is
[`number-field-sieve.md`](number-field-sieve.md). Here we'll just use everyday
analogies. No background needed — if you can multiply two numbers, you're ready.

---

## What does this program actually do?

It takes a gigantic number and finds which two numbers were multiplied together
to make it. That's it. That's called **factoring**.

Sounds simple? Try it the other way. Multiplying is easy:

> 61 × 53 = 3233

Now cover up the "61 × 53" and look at just **3233**. Which two numbers multiply
to give it? You'd have to *search*. For a small number you'll get there. But the
numbers this program targets have **hundreds of digits** — and for those,
"just search" would take every computer on Earth longer than the universe has
existed.

Multiplying is like **stirring two colors of paint together**: trivial. Figuring
out exactly which two colors went in, just by staring at the mix:
nearly impossible. Factoring is un-stirring the paint.

---

## Why should anyone care?

Because that difficulty is a **lock that protects almost everything online**.

When you see the little padlock in your browser, or use your bank's app, or send
a private message, much of that security rests on a method called **RSA**. RSA
deliberately builds a public number by multiplying two enormous secret primes
together. Anyone can see the big number; only someone who can *factor* it back
into its two pieces can break the lock. The whole system works **only because
factoring is so hard**.

So a program like this is two things at once:

- a **research tool** for mathematicians studying how hard factoring really is, and
- a **stress test** that tells the world how big your secret numbers need to be
  to stay safe.

It is the friendly, in-the-open version of the thing every security system is
betting against.

---

## The one clever idea at the heart of it

Here's the trick everything else is built around. Suppose I can find **two
different numbers that, when squared, land on the same place** (as far as our big
number is concerned). Picture two keys cut to different shapes that nonetheless
open the same lock. The fact that two *different* keys fit means the lock has a
hidden weakness — and that weakness *is* the factor.

There's a grade-school version you can actually check. Notice that

> (a − b) × (a + b) = a² − b²

In words: the gap between two perfect squares always splits neatly into two
pieces. So if we can write our target number as **one square minus another
square**, we've factored it for free. Let's do it for real with **5959**:

- The nearest square above 5959 is 6400 = 80².
- Subtract: 6400 − 5959 = **441**.
- And 441 is itself a perfect square: 441 = 21².

So 5959 = 80² − 21² = (80 − 21) × (80 + 21) = **59 × 101**. ✓

Go ahead, check it: 59 × 101 = 5959. We just factored a number by turning it into
a difference of two squares. That "two squares" idea is the seed of the entire
method. The catch: for a 200-digit number you'll never *stumble* onto the right
pair of squares by luck. The rest of this program is an industrial machine for
**manufacturing** that lucky pair on purpose.

---

## Step one: hunt for "nice" numbers

The machine can't build the magic squares directly. Instead it collects millions
of small **clues** and later snaps them together.

A clue is a number that happens to be **"smooth."** A smooth number is one built
entirely out of *small* prime building blocks — think of it as a LEGO model made
only from tiny, common bricks (2, 3, 5, 7, 11, …) with no big weird custom piece
in the middle. For example 360 = 2×2×2×3×3×5 is smooth; a number that needs a
giant prime like 9,999,991 is not.

Smooth numbers are gold because they're **easy to work with** — you can track
exactly which small bricks they're made of. The whole hunt is really a hunt for
smooth numbers.

How do you find them in a sea of billions of candidates without testing each one
the slow way? With a **sieve** — the same idea as the ancient "Sieve of
Eratosthenes" you may have met in school for finding primes. Rather than
examining every number individually, you sweep through the whole range over and
over, each pass cheaply marking off multiples of one small brick. After all the
sweeps, the numbers that got marked the most are your smooth nuggets. It's
**panning for gold**: you don't inspect each grain of gravel; you sift the whole
riverbed at once and let the nuggets reveal themselves. This sifting is the
longest, most computer-hungry part of the job — and it splits beautifully across
many processor cores, which is why more cores means faster results.

---

## The genuinely clever leap (told gently)

If you just sieved ordinary numbers, you'd be doing the older, slower "Quadratic
Sieve." The **Number Field Sieve** — the method this program is named after —
adds one brilliant twist that makes it the fastest known.

The twist: it **reshapes the problem so the numbers it has to test are much
smaller**. And smaller numbers are *far* more likely to be smooth (small LEGO
models are easier to build from small bricks). Fewer giant numbers to inspect,
more nuggets per scoop.

To pull this off, mathematicians step **outside ordinary numbers** into a larger,
custom-built number system — a kind of *parallel world of numbers*. You've
actually seen this trick before: engineers routinely use "imaginary" numbers
(involving the square root of −1) to solve very real problems about electricity
and waves, then translate the answer back to the real world at the end. The
Number Field Sieve does the same kind of thing: it pops into a roomier number
world where the factoring puzzle is easier to chip away at, does its work there,
and carries the result back home.

**You do not need to understand that parallel world to get the big picture.** The
only thing that matters is *why* it's worth the trouble: it shrinks the numbers
being tested, and shrinking them is what buys all the speed.

Setting up the *right* parallel world for a given target is a craft in itself —
it's the program's first job, and a good choice can make the whole computation
several times faster. It's like scouting the best fishing spot and choosing the
right bait before you ever cast a line.

---

## Step two: snap the clues together

After the long sieving hunt, the program is holding a mountain of clues — far
more than it needs, many of them junk or duplicates. So it **tidies the pile**:
throws out duplicates, and discards any clue that mentions a brick *no other clue
mentions* (a brick that shows up only once can never be paired off, so that clue
is useless). Tidying one clue can make another become useless, so it sweeps
through again and again until only good, interlocking clues remain.

Now comes the puzzle. Remember the goal: build a perfect square. A number is a
perfect square exactly when **every one of its building blocks comes in an even
number of copies** — every brick paired up, none left over. Each clue contributes
some bricks. The program must choose a **subset of clues whose bricks all pair up
evenly** when combined.

Think of it as a colossal **"Lights Out" puzzle**, or balancing a giant chemical
equation: each clue flips certain switches on or off, and you're looking for the
exact combination of clues that returns *every* switch to "off." With millions of
clues and millions of switches, you can't eyeball it — but it's a very organized
kind of bookkeeping, and computers are superb at it (this is the step
mathematicians call the "linear algebra," done with specialized large-scale
methods so the giant-but-mostly-empty puzzle stays manageable).

The payoff of solving the puzzle: a set of clues that multiply together into a
**perfect square** — and, by the magic of the parallel-world setup, into a
perfect square in **two different ways at once.** Which is exactly the "two keys,
one lock" situation we wanted from the very start.

---

## Step three: open the lock

With the two squares in hand, the program takes their square roots and compares
them. Because the two roots are different keys that fit the same lock, a quick,
classic calculation (finding what they "share" with the target — a
greatest-common-divisor, the same thing you did with fractions in school, just on
giant numbers) pops out a **real factor** of the original number.

Occasionally the puzzle hands back a dud — the two keys turn out to be the *same*
key, which tells you nothing. No problem: the puzzle-solving step produces several
independent solutions, so the program just tries the next one. About half of them
work, so success is quick.

And that's the whole machine:

| Stage | In plain terms | What it's like |
|---|---|---|
| 1. Choose the setup | pick the best "parallel world" for this number | scouting the fishing spot |
| 2. Sieve | hunt millions of smooth "clue" numbers | panning for gold (the long part) |
| 3. Tidy up | drop duplicates and dead-end clues | sorting the catch |
| 4. Solve the puzzle | combine clues so everything pairs evenly | a giant Lights-Out puzzle |
| 5. Open the lock | take square roots, compute the shared factor | the lock springs open |

---

## So how long does this take?

This is the **fastest general method humanity knows**, and it's still slow on
purpose-built numbers. Each time the target grows by a handful of digits, the
work roughly multiplies. A 60-digit number falls in well under a minute on a good
desktop; a 90-digit number in a few minutes; but the difficulty climbs so steeply
that the ~600-digit numbers guarding serious secrets stay out of reach of *all*
the world's computers combined — for now. (You can see the curve start to bend on
real hardware in [`../BENCHMARKS.md`](../BENCHMARKS.md).)

That "for now" carries one famous asterisk: a future **quantum computer** running
a method called *Shor's algorithm* could factor those numbers quickly, which is
why cryptographers are already moving to new "post-quantum" locks. Until such
machines exist at scale, the difficulty measured by tools like this one is what
keeps today's encryption standing.

---

## Want to see the real thing?

Everything above has a precise mathematical counterpart — perfect squares modulo
a number, "smooth" factorizations, the parallel-world *number fields*, the
switch-flipping puzzle as linear algebra over a two-value arithmetic, and the
final square roots. If you're curious how it's really done, the rigorous version
is right next door:
[**`number-field-sieve.md`**](number-field-sieve.md).

---

*Part of [cado-nfs-2.3.1-modern](https://github.com/doublegate/cado-nfs-2.3.1-modern).
The clever algorithm described here is the work of the upstream CADO-NFS team
(see [`../AUTHORS`](../AUTHORS)); this page is just a friendly tour of it.*
