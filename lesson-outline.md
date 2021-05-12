---
layout: page
title: Running HemeLB on HPC Systems -  Lesson Outline
---

## How to use this outline 

The following list of items is meant as a guide on what content should go where in this repo. 
This should work as a guide where you can contribute. If a bullet point is prefixed by a
file name, this is the lesson where the listed content should go into. This document is
meant as a concept map converted into a flow learning goals and questions.

## Accelerating HemeLB on a HPC

* []

* [index.md: Prelude](index.md):Why should I take this course?
    * Why should I bother about software performance?
    * What can I expect to learn from this course?

* [01-benchmarking-and-scaling.md:](_episodes/01-benchmarking-and-scaling.md): Brief notes
  on software performance
    * What is software performance?
    * Why is software performance important?
    * How can performance be measured?
    * What is meant by flops, walltime and CPU hours?
    * What can affect performance?

* [02-hemelb-bottlenecks.md: What are the common limitations to getting peak performance](_episodes/02-hemelb-bottlenecks.md):
  about benchmark and scaling
    * What is benchmarking?
    * What are the factors that can affect a benchmark?
        * _**Case study 1:**_ A simple benchmarking example of LAMMPS in a HPC
        * _**Hands-on 1:**_ Can you do it on your own?
    * What is scaling?
    * How do I perform scaling analysis?
    * Quntifying speedup: t<sub>1</sub>/t<sub>p</sub>
    * Am I wasting my resourse?
        * _**Case Study 2:**_ Get scaling data for a LAMMPS run
        * _**Hands-on 2:**_ Do a scaling analysis

* [03-accelerating-hemelb.md](_episodes/03-accelerating-hemelb.md)

* [04-invoking-packages-in-hemelb.md](_episodes/04-invoking-pacakges-in-hemelb.md)

* [05-gpus-with-hemelb.md](_episodes/05-gpus-with-hemelb.md)