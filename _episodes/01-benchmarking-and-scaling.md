---
title: "Benchmarking and Scaling"
teaching: 15
exercises: 15
questions:
- "What is benchmarking?"
- "How do I do a benchmark?"
- "What is scaling?"
- "How do I perform a scaling analysis?"
objectives:
- "Be able to perform a benchmark analysis of an application"
- "Be able to perform a scaling analysis of an application"
keypoints:
- "Benchmarking is a way of assessing the performance of a program or set of programs"

---

## What is benchmarking?

To get an idea of what we mean by benchmarking, let’s take the example of a sprint athlete. 
The athlete runs a predetermined distance on a particular surface, and a time is recorded. Based on different
conditions, such as how dry or wet the surface is, or what the surface is made of (grass, sand, or track) the times
of the sprinter to cover a distance (100m, 200m, 400m etc) will differ. If you know where the sprinter is running,
and what the conditions were like, when the sprinter sets a certain time you can cross-correlate it with the known 
times associated with certain surfaces (our benchmarks) to judge how well they are performing.

Benchmarking in computing works in a similar way: it is a way of assessing the performance of a program (or set of
programs), and benchmark tests are designed to mimic a particular type of workload on a component or system. They can
also be used to measure differing performance across different systems. Usually codes are tested on different computer
architectures to see how a code performs on each one. Like our sprinter, the times of benchmarks depends on a number
of things: software, hardware or the computer itself and it’s architecture.

> ## Running a HemeLB job on your HPC
> 
> **EDITME** Use 10 core example for bifurcation, 5000 steps
>
> Put the job script, input files and outputs into a folder called `10c-bif`.
>
{: .challenge}

## Understanding your output files

**EDITME** How to understand your output files, and gain understanding of where most time is spent. Use examples.

> ## Editing the submission script
>
> Make a directory called `2n-bif` and copy the input files and job script into used in the previous exercise into it.
> 
> Modify this copied submission script, editing it in such a way to include multiple nodes.
>
{: .challenge}

> ## Callout: Local vs system-wide installations
>
> Whenever you get access to an HPC system, there are usually two ways to get access to
> software: either you use a system-wide installation or you install it yourself. For widely
> used applications, it is likely that you should be able to find a system-wide installation.
> In many cases using the system-wide installation is the better option since the system
> administrators will (hopefully) have configured the application to run optimally for
> that system. If you can't easily find your application, contact user support for the
> system to help you.
>
> You should still check the benchmark case though! Sometimes system administrators are short
> on time or background knowledge of applications and do not do thorough testing.
{: .callout}

## Benchmarking in HemeLB: A case study

In the next section we will look at how we can use all this information to perform a scalability study, but first
let us overview the concepts of benchmarking.

> ## The ideal benchmarking study
>
> Benchmarking is a process that judges how well a piece of software runs on a system. Based on what you have learned
> thus far from running your own benchmarks, which of the following would represent a good benchmarking analysis?
>
> **EDITME** Add graphs CW
>
> 1. Linear increase in core count to 64 cores (x-axis) ~10 points
> 2. Increase by x2 by core count up to 2048 cores (x-axis), 12 points
> 3. Increase by x2 by core count up to 131072 cores (x-axis), 18 points
> 4. Increase by 10 cores up to 2000 cores (200 points)
> 5. Linear increase 1-20 nodes ~20 points
> 6. **EDITME** Another wrong answer
> 
>
> > ## Solution
> > 
> > 1. No, the core counts that are being benchmarked are too low and the number of points is not sufficient
> > 2. Yes, but depends on the software you are working with, how you want to use it, and how big your system is.
> >    This does not give a true view how scalable it is at higher core counts.
> > 3. Yes. If the system allows it and you have more cores at your disposal, this is the ideal benchmark to run. But
> >    as with #2, it depends on how you wish to utlise the software.
> > 4. No, although it increases by a factor of 10 to 2000 cores, there are too many points on the graph and therefore
> >    would be highly impractical. Benchmarks are used to get an idea of scalability, the exact performance will vary
> >    with every benchmark run.
> > 5. Yes. This is also a suitable metric for benchmarking, similar to response #3.
> >
> {: .solution}
{: .challenge}

## Scaling 

Going back to our athelete example from earlier, we may have determined the conditions and done a few benchmarks on
his/her performance over different distances, we might have learned a few things.

- how fast the athelete can run over short, medium and long distances
- the point at which the athelete can no longer perform at peak performance

In computational sense, scalability is defined as **the ability to handle more work as the size of the computer**
**application grows**. This term of scalability or scaling is widely used to indicate the ability of hardware and
software to deliver greater comptational power when the amount of resources is increased. When you are working on an
HPC cluster, it is very important that it is scalable, i.e. that the performance doesn't rapidly decrease the more 
cores/nodes that are assigned to a series of tasks.

Scalability can also be looked as in terms of parallelisation efficiency, which is the ratio between the actual
speedup and the ideal speedup obtained when using a certain number of processes. The overall term of speedup in HPC
can be defined with the formula `Speedup = t(1)/t(N)`.

Here, `t(1)` is the computational time for running the software using one processor and `t(N)` is the comptational time
running the software with N proceeses. An ideal situation is to have a linear speedup, equal to the number of
processors (speedup = N), so every processor contributes 100% of its computational power. In most cases, as an
idealised situation this is very hard to attain.

### Weak scaling vs Strong scaling

Applications can be divided into either **strong scaling** or **weak scaling** applications.

For **weak scaling**, the problem size increases as does the number of processors. In this situation, we usually want
to increase our workload without increasing our *walltime*, and we do that by using additional resources.

> ## Gustafson-Barsis' Law
>
> *Speedup should be measured by scaling the problem to the number of processes, not by fixing the problem size.*
> 
> `Speedup = s + p * N`
>
> where `s` is the proportion of the execution time spent on serial code, `p` is the amount of time spent on 
> parallelised code and `N` is the number of processes.
>
{: .callout}

For **strong scaling**, the number of processes is increased whilst the problem size remains the same, resulting in a
reduced workload for each processor. 

> ## Amdahl's Law
>
> The speedup is limited by the fraction of the seria part of the software that is not amenable to parallelisation
>
> `Speedup = 1/( s + p / N )`
> 
> where `s` is the proportion of the execution time spent on serial code, `p` is the amount of time spent on 
> parallelised code and `N` is the number of processes.
>
{: .callout}

Whether one is dealing with a strong or weak scaling 

> ## Determine best performance from a scalability study
> 
> Consider the following scalability plot for a random application
> ~~~
> |          /
> |         /
> |        /
> |       /  
> |      / b
> |     / _._
> |   a/_/   \c
> |   //
> |  //
> |_//_________
> ~~~
> {: .language-bash}
> #processors/nodes
> 
> At what point would you consider to be peak performance in this example.
>
> - 1. a
> - 2. b
> - 3. c
> - 4. None of the above 
> 
> You may find that this graph would differ if you ran the same code on a different machine. Why?
> 
> > # Solution
> > 
> > 1. No, the performance is still increasing, at this point we are no longer achieving perfect scalability.
> > 2. Yes, the performance peaks at this location, and one cannot get higher speed up with this set up.
> > 3. No, peak performance has already been achieved, and increasing the core count will onlt reduce performance.
> > 4. No, although you can run extra benchmarks to find the exact number of cores at which the inflection point truly
> >    lies, there is no real purpose for doing so.
> >
> > Tying into the answer for #4, if you produce scalability studies on different machines, they will be different
> > because of the different setup, hardware of the machine. You are never going to get two scalability studies which
> > are identical, but they will agree to some point.
> {: .solution}
{: .challenge}

**EDITME** (This section needs more expandid)

Scaling behaviour in computation is centred around the effective use of resources as you
scale up the amount of computing resources you use. An example of "perfect" scaling would
be that when we use twice as many CPUs, we get an answer in half the time. "Bad" scaling
would be when the answer takes only 10% less time when we double the CPUs. This example
is one of **strong scaling**, where the workload doesn't change as we increase our
resources.

> ## Plotting strong scalability
>
> Use the original job script for 2 nodes and run it.
>
> Now that
> you have results for 1 core, 4 cores and 2 nodes, create a *scalability plot* with
> the number of CPU cores on the X-axis and the loop times on the Y-axis (use your
> favourite plotting tool, an online plotter or even pen and paper).
>
> Are you close to "perfect" scalability?
>
{: .challenge}

### Weak scaling

For **weak scaling**, we want usually want to increase our workload without increasing
our *walltime*,
and we do that by using additional resources. To consider this in more detail, let's head
back to our chefs again from the previous episode, where we had more people to serve
but the same amount of time to do it in.

We hired extra chefs who have specialisations but let us assume that they are all bound
by secrecy, and are not allowed to reveal to you
what their craft is, pastry, meat, fish, soup, etc. You have to find out what their
specialities are, what do you do? Do a test run and assign a chef to each course. Having
a worker set to each task is all well and good, but there are certain combinations which
work and some which do not, you might get away with your starter chef preparing a fish
course, or your lamb chef switching to cook beef and vice versa, but you wouldn't put
your pastry chef in charge of the main meat dish, you leave that to someone more
qualified and better suited to the job.

Scaling in computing works in a similar way, thankfully not to that level of detail
where one specific core is suited to one specific task, but finding the best combination
is important and can hugely impact your code's performance. As ever with enhancing
performance, you may have the resources, but the effective use of the resources is
where the challenge lies. Having each chef cooking their specialised dishes would be
good weak scaling: an effective use of your additional resources. Poor weak scaling
will likely result from having your pastry chef doing the main dish.
