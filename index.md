---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # only page that doesn't follow the pattern /:path/index.html
---

HemeLB is a 3D blood flow simulation code based on the lattice Boltzmann method. It is an open-source code built
using C++ and MPI and has demonstrated excellent scaling performance on some of the largest and fastest supercomputers
on the planet. One particular challenge for simulating the typically sparse domains characteristic of blood vessels is
dealing with the sparse domain space - for a bounding box of a given domain maybe 1% (and often much less) actually 
consists of fluid that you are interested in studying. During its development, HemeLB has been specifically optimised 
to efficiently study such domains. The full feature version of HemeLB can be found [here](https://github.com/hemelb-codes/).
However for this lesson, we recommend using the HemePure example - a version of HemeLB that has
further optimisations for scalable simulation on CPU based machines.

This workshop is specifically aimed at running HemeLB on an HPC system. You may be running HemeLB on either a desktop,
laptop or already on an HPC system, however ineffective use of HemeLB can lead to running jobs for
longer than necessary. Being able configure HemeLB on an HPC system effectively can speed up simulations significantly
an improve it's performance. This workshop will look to address these issues.

Some questions that you may ask yourself are;

* What is meant by the term ***performance*** in relation to piece of software?
* How do I measure performance?
* How can I know the expected performance of a piece of software?
* How do I compare HemeLB running on my HPC to its expected performance?
* **If software performance is not optimal in my system, is there something that can I
  do to accelerate it?**

If you have asked the any of above questions, then you might be a good candidate for
taking this course.

An HPC system is a complex computing platform that usually has several hardware
components. Terms that might be familiar are CPU, RAM and GPU since you can find these
in your own laptop or server. There are other commonly used terms such as "shared
memory", "distributed computing", "accelerator", "interconnect" and "high performance
storage" that may be a little less familiar. In this course we will try to cover the
subset of these that are relevant to your use case with HemeLB.

On any HPC system with a variety of hardware components, software performance will vary
depending on what components it is using, and how optimized the code is for those
components. There are usually no such complications on a standard desktop or laptop,
running on an HPC is very, very different.

> ## Note
>
> - This is the draft HPC Carpentry release. Comments and feedback are welcome.
>
{: .callout}

> ## Prerequisites
>
> - Basic experience with working on an HPC system is required. If you are new to these
>   these types of systems we recommend participants to go through the
>   [Introduction to High-Performance Computing](https://hpc-carpentry.github.io/hpc-intro/)
>   from [HPC Carpentry](https://hpc-carpentry.github.io/).
> - You should have some familiarity with the concepts behind MPI and OpenMP as they are useful
>   tools for benchmarking and scalability studies.
> - You should be familiar with working with HemeLB, how to install it and how to
>   [run a basic HemeLB simulation](http://hemelb.org.s3-website.eu-west-2.amazonaws.com/tutorials/simulation/). 
>   For running on HPC systems and submitting a bash script, you can refer to the HemeLB
>   [documentation](https://github.com/hemelb-codes/hemelb/raw/main/Doc/hemelb_documentation.doc)
>
{: .prereq}
