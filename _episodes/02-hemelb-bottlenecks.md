---
title: "Bottlenecks in HemeLB"
teaching: 40
exercises: 45
questions:
- "How can I identify the main bottlenecks in HemeLB?"
- "How do I come up with a strategy for finding the best optimisation?"
- "What is load balancing?"
objectives:
- "Learn how to analyse timing data in HemeLB and determine bottlenecks"
keypoints:
- "The best way to identify bottlenecks is to run different benchmarks on a smaller system and
  compare it to a representative system"
- "Effective load balancing is being able to distribute an equal amount of work across
  processes"
usemathjax: true
---

- How is HemeLB parallelised?

Before using any accelerator package to speedup your runs, it is always wise to identify
performance *bottlenecks*. The term "bottleneck" refers to specific parts of an
application that are unable to
keep pace with the rest of the calculation, thus slowing overall performance.

Therefore, you need to ask yourself these questions:
* Are my runs slower than expected?
* What is it that is hindering us getting the expected scaling behaviour?

## Identify bottlenecks

Identifying (and addressing) performance bottlenecks is important as this could save you
a lot of
computation time and resources. The best way to do this is to start with a reasonably
representative system having a modest system size and run for a few hundred/thousand
timesteps.

- Does HemeLB have a timing breakdown table?

Based on this breakdown table we will work on a few examples and try to
understand how to identify bottlenecks from this output. Ultimately, we will try to find
a way to minimise the walltime by adjusting ****EDITME (what?)****

> ## Important!
> For many of these exercises, the exact modifications to job scripts that you will need
> to implement are system
> specific. Check with your instructor or your HPC institution's helpdesk for information specific
> to your HPC system.
>
{: .callout}

> ## Example timing breakdown for small system
> Using your previous job script for a a *serial* run (i.e. on a single core), replace
> the input file with the one for the **small** system and run it on the HPC system.
>
> Take a look at the resulting timing breakdown table and discuss with your neighbour
> what you think you should target to get a performance gain.
>
> > ## Solution
> > 
> {: .solution}
{: .challenge}

## Effects due to system size on resource used

Different sized systems might behave differently
as we increase our resource usage since they will have different distributions of work
among our available resources.

> ## Analysing the small system
>
> Below is an example timing breakdown for 4000 atoms LJ-system with 40 MPI ranks
>
> ~~~
> MPI task timing breakdown:
> Section |  min time  |  avg time  |  max time  |%varavg| %total
> ---------------------------------------------------------------
> Pair    | 0.24445    | 0.25868    | 0.27154    |   1.2 | 52.44
> Neigh   | 0.045376   | 0.046512   | 0.048671   |   0.3 |  9.43
> Comm    | 0.16342    | 0.17854    | 0.19398    |   1.6 | 36.20
> Output  | 0.0001415  | 0.00015538 | 0.00032134 |   0.0 |  0.03
> Modify  | 0.0053594  | 0.0055818  | 0.0058588  |   0.1 |  1.13
> Other   |            | 0.003803   |            |       |  0.77
> ~~~
> {: .output}
>
> Can you discuss any observations that you can make from the above table? What could
> be the rationale behind such a change of **EDITME something**
>
> > ## Solution
> >
> {: .solution}
{: .discussion}

> ## Analysing the large system
>
> > ## Solution
> >
> {: .solution}
{: .discussion}

## Scalability

Since we have information about the timings for different components of the calculation,
we can perform a scalability study for each of the components.

> ## Investigating scalability on a number of nodes
>
> > ## Solution
> > 
> {: .solution}
{: .challenge}

## MPI vs OpenMP

- **EDITME - bottleneck identified, now how do you implement solving it**

Let us discuss a few situations:


Let us now build some hands-on experience to develop some feeling on how this works.

> ## Case study: System 1
>
{: .callout}

## Load balancing
One important issue with MPI-based
parallelization is that it can under-perform for systems with inhomogeneous
distribution of particles, or systems having lots of empty space in them. It is pretty
common that the evolution of simulated systems evolve over time to reflect such a case.
This results in *load imbalance*. While some of the processors are assigned with
finite number of
particles to deal with for such systems, a few processors could have far less atoms (or
none) to do any calculation and this results in an overall loss in parallel efficiency.
This situation is more likely to expose itself as you scale up to a large
large number of processors.


> ## Example timing breakdown for system
>
> > ## Solution
> > 
> {: .solution}
{: .discussion}
