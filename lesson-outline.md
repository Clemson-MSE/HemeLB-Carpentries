---
layout: page
title: Running HemeLB on HPC Systems -  Lesson Outline
---

## Main points to remember

1. The Carpentries style works by backward design;
    - Identify practical skills to teach
    - Design challenges to give opportunity for learners to practice and integrate skills
    - Identify what needs to be taught for learners to acquire the skills 

2. Those that wish to learn HemeLB will undoubtedly find weird (and sometimes completely bizarre) 
   ways of doing things. So, be aware of 
   [expert awareness gap](https://carpentries.github.io/instructor-training/03-expertise/index.html#expertise-and-teaching)
   
3. Ensure that you fork the repository to your own GitHub account. Be certain that you are working in 
   `gh-pages` branch. From there you can work locally and submit pull requests.

4. Tests have been implemented with the repo, and should run once commit is staged on GitHub or pushed
   from local machine. These checks include spelling, and page build checks.

5. To see how your page builds, head to; Settings -> Pages and see the link highlighted in green. It will 
   be of the form `https://{youraccount}.github.io/HemeLB-Carpentries`. Copy and paste it into About on the
   left hand side of the main screen of your forked repo to see how it turns out. The main repo page build is located
   [here](https://hemelb-dev.github.io/HemeLB-Carpentries/).

6. Chris will review material before merging your forked repo with the 
   [main repo](https://github.com/HemeLB-dev/HemeLB-Carpentries), and can assist with rewording of material if
   too dense

4. To bring your own fork up to date with main repo;
    - `git fetch upstream`
    - `git checkout gh-pages`
    - `git merge upstream/gh-pages`

## Lesson Outline

* [index.md: Prelude](index.md): 
    * CW
    * Why should I bother about software performance?
    * What can I expect to learn from this course?

* [01-benchmarking-and-scaling.md: How to do a benchmarking and scaling analysis](_episodes/01-benchmarking-and-scaling.md): 
    * What is benchmarking?
        - worth expanding on [previous work](https://fzj-jsc.github.io/tuning_lammps/03-benchmark-and-scaling/index.html)
    * What are the factors that can affect a benchmark?
    * How can I perform a benchmark in HemeLB
    * What is scaling?
    * How can I perform a scaling analysis with HemeLB?
        - Use examples of the +300k cores, how was it done?
        - [EuroHPC Abstract 2020](https://events.prace-ri.eu/event/1018/sessions/3644/attachments/1531/2780/EuroHPC2020_UpdatedAbstract_JMcCullough.pdf)
        - [POP-COE](https://pop-coe.eu/blog/190x-strong-scaling-speed-up-of-hemelb-simulation-on-supermuc-ng)
    * (Maybe explanation of MPI/OpenMP (CW))


* [02-hemelb-bottlenecks.md: What are the common limitations to getting peak performance](_episodes/02-hemelb-bottlenecks.md):
    * Load imbalancing (see slide [84](https://drive.google.com/file/d/1ZVhmfIC9lPhjTIxgnViKuH3NNAq_HfSt/view?usp=sharing))
        - what do beginners need to be aware of (flags to be enabled/disabled?)
        - the effects of geometry on performance [2013 paper](https://www.sciencedirect.com/science/article/pii/S1877750313000240?via%3Dihub)
    * Bottlenecks in one simulation type compared to another
        - What bottlenecks were encountered [here](https://www.compbiomed.eu/compbiomed-webinar-10/), and 
          how were they overcome?
    * What componetns of built in practices slow simulations down considerably/common pitfalls which, when avoided can
      greatly enhance speedup
    * Visualisation of output, can generation of this lead to bottlenecks?

* [03-accelerating-hemelb.md](_episodes/03-accelerating-hemelb.md)
    * Packages which can be used with HemeLB, 

* [04-invoking-packages-in-hemelb.md](_episodes/04-invoking-pacakges-in-hemelb.md)

* [05-gpus-with-hemelb.md](_episodes/05-gpus-with-hemelb.md)

## Timeline

- May 24th 
  - discussion of existing material
  - reach consensus/finalise workplan (excluding GPU material) as outlined in `lesson-outline.md`
  - Plan Sprint 1 May 31st - June 14th
    - 01-benchmarking-and-scaling - exercises + content
    - 02-hemelb-bottlenecks - exercises

- May 24-31 add material

- May 31st
  - Start Sprint 1

- June 14
  - Start Sprint 2
    - 02-hemelb-bottlenecks - content
    - 04-invoking-packages-in-hemelb - explore possibilities

- June 28
  - Episodes 01, 02, finished
  - Start Sprint 3
    - 03-accelerating-hemelb - exercises + content


- July 12
  - Episode 03 exercises finished
  - Start Sprint 4
    - 03-accelerating-hemelb - content
    - 04-invoking-pacakges-with-hemelb remain

- July 26
  - Start Sprint 5
    - Finish ep 3,4

- August 9
  - LEAST PUBLISHABLE UNIT (episodes 1,2,3,4)
  - Iron out final details
  - Revisit GPU (episode 5)

- August 23
  - Finalise GPU material