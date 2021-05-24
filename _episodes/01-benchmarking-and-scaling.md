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

- **EDITME** (Insert information on benchmarking. This can be copied from the LAMMPS material or modified accordlingly.
  Ideally expanded upon given the nature of the library.)

## Case study: Benchmarking with HemeLB

As an example, let's create some benchmarks for you to compare the performance of some
'standard' systems in HemeLB. Using a 'standard' system is a good idea as a first attempt, since
we can measure our benchmarks against (published) information from others.  Knowing that
our installation is "sane" is a critical first step before we embark on generating **our
own benchmarks for our own use case**.

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
> You should still check the benchmark case though! Sometimes administrators are short
> on time or background knowledge of applications and do not do thorough testing.
{: .callout}

> ## Exercise 1: Running a HemeLB job on an HPC system
>
> Can you list the bare minimum files that you need to schedule a HemeLB job on an HPC
> system?
>
> > ## Solution
> > For running a HemeLB job, we need:
> > 1. File 1 
> > 2. File 2
> >
> > **EDITME** Additional explanations
> {: .solution}
{: .challenge}

The input file we need for the LJ-system is reproduced below:

**EDITME**

~~~
included file
~~~
{: .source}

- How to go about making a job file for HemeLB (include starting point and explanation
  of each steps)

~~~
included from snippets
~~~
{: .language-bash}

> ## Exercise 2: Edit a submission script for a HemeLB job
>
> Duplicate the job script we just created so that we have versions that will run on
> 1 core and 4 cores.
>
>
> > ## Solution
> >
> {: .solution}
{: .challenge}

### Understanding the output files

- Any information about versions, OMP, MPI variables

Useful keywords to search for include:

  * **`EDITME...`** 


> ## Exercise 3: Now run a benchmark...
>
> From the jobs that you ran previously,
> extract the loop times for your runs and see how they compare
> with the HemeLB standard benchmark and with the performance for two other HPC systems.
>
> | HPC system | 1 core (sec) | 4 core (sec) |
> |----------- | ------------ |------------- |
> | HemeLB     | 2.26185      | 0.635957     |
> | HPC 1      | 2.24207      | 0.592148     |
> | HPC 2      | 1.76553      | 0.531145     |
> | MY HPC     |     ?        |     ?        |
>
> Why might these results differ?
>
{: .challenge}

## Scaling 

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
