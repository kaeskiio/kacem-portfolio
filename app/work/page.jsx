"use client";

import { motion } from "framer-motion";
import React, { useState } from "react";

import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";

import { BsArrowUpRight, BsGithub } from "react-icons/bs";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import Link from "next/link";
import Image from "next/image";
import WorkSliderBtns from "@/components/WorkSliderBtns";

const projects = [
  {
    num: "01",
    category: "Chemistry Memory",
    title: "project 1",
    description:
      "This website provides chemistry students with effective and interactive practice tools that make memorization not only achievable but also enjoyable.",
    stack: [{ name: "Html" }, { name: "CSS" }, { name: "Javascript" }],
    image: "/assets/work/chem.png",
    live: "https://chemistryquiz.net/",
    github: "https://github.com/kacemettahali/ChemistryWeb",
  },
  {
    num: "02",
    category: "Shadow Creek Tennis Camp",
    title: "project 2",
    description:
      "This website is dedicated to promoting Shadow Creek Summer Tennis Camp programs and aimed at fostering a love for tennis and developing skills in players of all ages and levels.di.",
    stack: [{ name: "HTML" }, { name: "CSS" }, { name: "Javascript" }],
    image: "/assets/work/tennis.jpg",
    live: "https://www.sctenniscamp.org/",
    github: "https://github.com/kacemettahali/TennisWeb",
  },
  {
    num: "03",
    category: "OwlPhysiomonitor",
    title: "project 3",
    description:
      "The website provides underserved populations with the opportunity for quality healthcare and aid.",
    stack: [{ name: "HTML" }, { name: "CSS" }, { name: "Javascript" }],
    image: "/assets/work/health.jpg",
    live: "https://www.OwlPhysiomonitor.com",
    github: "https://github.com/kacemettahali/YoungScholars",
  },
  {
    num: "04",
    category: "Weather App UI",
    title: "project 4",
    description:
      "A sleek and intuitive interface for checking current weather conditions and forecasts. Features include temperature readings and weather icons for easy navigation.",
    stack: [{ name: "Swift" }],
    image: "/assets/work/weather.jpg",
    live: "",
    github: "https://github.com/kacemettahali/WeatherApp",
  },
  {
    num: "05",
    category: "Computer Vision Game",
    title: "project 5",
    description:
      "Guess the hidden image as random pixels are gradually revealed. Identify the image with as few revealed pixels as possible. Models Computer Vision.",
    stack: [{ name: "Python" }],
    image: "/assets/work/vision.jpg",
    live: "",
    github: "https://github.com/kacemettahali/computerVisionGame",
  },
  {
    num: "06",
    category: "IPhone 15 Pro Web",
    title: "project 6",
    description:
      "This project is a replica of Apple's iPhone 15 Pro Titanium(my phone) website.",
    stack: [{ name: "React.js" }, {name: "Tailwind.css"}, {name: "GSAP"}, {name: "Three.js"}, {name: "Vite"}],
    image: "/assets/work/iphone.jpeg",
    live: "",
    github: "https://github.com/kacemettahali/IPhone-15Pro-Web",
  },
];

const Work = () => {
  const [project, setProject] = useState(projects[0]);

  const handleSlideChange = (swiper) => {
    // get current slide index
    const currentIndex = swiper.activeIndex;
    // update project state based on current slide index
    setProject(projects[currentIndex]);
  };

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{
        opacity: 1,
        transition: { delay: 2.4, duration: 0.4, ease: "easeIn" },
      }}
      className="min-h-[80vh] flex flex-col justify-center py-12 xl:px-0"
    >
      <div className="container mx-auto">
        <div className="flex flex-col xl:flex-row xl:gap-[30px]">
          <div className="w-full xl:w-[50%] xl:h-[460px] flex flex-col xl:justify-between order-2 xl:order-none">
            <div className="flex flex-col gap-[30px] h-[50%]">
              {/* outline num */}
              <div className="text-8xl leading-none font-extrabold text-transparent text-outline">
                {project.num}
              </div>
              {/* project category */}
              <h2 className="text-[42px] font-bold leading-none text-white group-hover:text-accent transition-all duration-500 capitalize">
                {project.category} 
              </h2>
              {/* project description */}
              <p className="text-white/60">{project.description}</p>
              {/* stack */}
              <ul className="flex gap-4">
                {project.stack.map((item, index) => {
                  return (
                    <li key={index} className="text-xl text-accent">
                      {item.name}
                      {/* remove the last comma */}
                      {index !== project.stack.length - 1 && ","}
                    </li>
                  );
                })}
              </ul>
              {/* border */}
              <div className="border border-white/20"></div>
              {/* buttons */}
              <div className="flex items-center gap-4">
                {/* live project button */}
                <Link href={project.live}>
                  <TooltipProvider delayDuration={100}>
                    <Tooltip>
                      <TooltipTrigger className="w-[70px] h-[70px] rounded-full bg-white/5 flex justify-center items-center group">
                        <BsArrowUpRight className="text-white text-3xl group-hover:text-accent" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Live project</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Link>
                {/* github project button */}
                <Link href={project.github}>
                  <TooltipProvider delayDuration={100}>
                    <Tooltip>
                      <TooltipTrigger className="w-[70px] h-[70px] rounded-full bg-white/5 flex justify-center items-center group">
                        <BsGithub className="text-white text-3xl group-hover:text-accent" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Github repository</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Link>
              </div>
            </div>
          </div>
          <div className="w-full xl:w-[50%]">
            <Swiper
              spaceBetween={30}
              slidesPerView={1}
              className="xl:h-[520px] mb-12"
              onSlideChange={handleSlideChange}
            >
              {projects.map((project, index) => {
                return (
                  <SwiperSlide key={index} className="w-full">
                    <div className="h-[460px] relative group flex justify-center items-center bg-pink-50/20">
                      {/* overlay */}
                      <div className="absolute top-0 bottom-0 w-full h-full bg-black/10 z-10"></div>
                      {/* image */}
                      <div className="relative w-full h-full">
                        <Image
                          src={project.image}
                          fill
                          className="object-cover"
                          alt=""
                        />
                      </div>
                    </div>
                  </SwiperSlide>
                );
              })}
              {/* slider buttons */}
              <WorkSliderBtns
                containerStyles="flex gap-2 absolute right-0 bottom-[calc(50%_-_22px)] xl:bottom-0 z-20 w-full justify-between xl:w-max xl:justify-none"
                btnStyles="bg-accent hover:bg-accent-hover text-primary text-[22px] w-[44px] h-[44px] flex justify-center items-center transition-all"
              />
            </Swiper>
          </div>
        </div>
      </div>
    </motion.section>
  );
};

export default Work;
