"use client";

import {
  FaHtml5,
  FaCss3,
  FaJs,
  FaReact,
  FaFigma,
  FaNodeJs,
  FaJava,
  FaGasPump,
  FaSwift,
  FaCalculator,
  FaSuperscript,
  FaFlask,
  FaPython,
  FaBook,
  FaCode,
  FaChartBar,
  FaDatabase,
  FaGit,
  FaGithub,
} from "react-icons/fa";

import { 
  SiTailwindcss, 
  SiNextdotjs, 
  SiScikitlearn,
  SiTensorflow, 
  SiPandas,
  SiNumpy,
} from "react-icons/si";




// about data
const about = {
  title: "About me",
  description:
    "I am a Computer Science student at Rice University with a concentration in Artificial Intelligence. I have extensive experience in AI/ML, full-stack development, and low-level programming. My work spans from mission-critical satellite algorithms at Lockheed Martin to cutting-edge AI research at MIT/Harvard, achieving significant performance improvements and building scalable systems.",
  info: [
    {
      fieldName: "Name",
      fieldValue: "Kacem Ettahali",
    },
    {
      fieldName: "Experience",
      fieldValue: "4+ Years",
    },
    {
      fieldName: "Email",
      fieldValue: "ke35@rice.edu",
    },
    {
      fieldName: "Languages",
      fieldValue: "English, Arabic",
    },
  ],
};

const volunteering = {
  title: "Volunteering",
  description:
    "I am actively involved in volunteering with organizations such as First Racquet where I organize multiple fundraisers and charity tournaments as the Executive VP of Directors donating 10 boxes consisting of 35 bags, 154 racquets, 270 articles of athleticwear, 52 pairs of shoes, 36 hats, 72 cans of tennis balls, and more, to Academia de Tenis Kikuxi Villas Club, a tennis academy in Luanda, Angola that shares the mission of First Racquet: empowering young Angolan athletes."
};

// experience data
const experience = {
  icon: "/assets/resume/badge.svg",
  title: "My Experience",
  description: "",
  items: [
    {
      company: "MIT & Harvard",
      position: "AI/ML & Full-Stack Software Engineer",
      duration: "May 2025 – Present",
    },
    {
      company: "Lockheed Martin",
      position: "Software Engineer",
      duration: "June 2025 – August 2025",
    },
    {
      company: "University of Texas at Austin",
      position: "Research Software Engineer",
      duration: "June 2024 – March 2025",
    },
    {
      company: "Rice University PATHS-UP Summer Program",
      position: "Instructor",
      duration: "2024",
    },
  ],
};

// skills data
const skills = {
  title: "My skills",
  description:
    "I am adept at utilizing many frameworks and languages across AI/ML, full-stack development, and low-level programming.",
  skillList: [
    {
      icon: <FaHtml5 />,
      name: "Html",
    },
    {
      icon: <FaCss3 />,
      name: "CSS",
    },
    {
      icon: <FaJs />,
      name: "Javascript",
    },
    {
      icon: <FaReact />,
      name: "React.js",
    },
    {
      icon: <SiNextdotjs />,
      name: "Next.js",
    },
    {
      icon: <SiTailwindcss />,
      name: "tailwind.css",
    },
    {
      icon: <FaPython />,
      name: "Python",
    },
    {
      icon: <FaNodeJs />,
      name: "Node.js",
    },
    {
      icon: <FaJava />,
      name: "Java",
    },
    {
      icon: <FaSwift />,
      name: "Swift",
    },
    {
      icon: <FaCode />,
      name: "C++",
    },
    {
      icon: <FaChartBar />,
      name: "R",
    },
    {
      icon: <FaDatabase />,
      name: "SQL",
    },
    {
      icon: <SiScikitlearn />,
      name: "Sci-kit Learn",
    },   
    {
      icon: <SiTensorflow />,
      name: "TensorFlow",
    },   
    {
      icon: <SiPandas />,
      name: "Pandas (Python)",
    },
    {
      icon: <SiNumpy />,
      name: "NumPy",
    },
    {
      icon: <FaFlask />,
      name: "Flask",
    },
    {
      icon: <FaGithub />,
      name: "Github",
    },
    {
      icon: <FaCalculator />,
      name: "Calculus",
    },
    {
      icon: <FaSuperscript />,
      name: "Linear Algebra",
    },
  ],
};

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { ScrollArea } from "@/components/ui/scroll-area";
import { motion } from "framer-motion";

const Resume = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{
        opacity: 1,
        transition: { delay: 2.4, duration: 0.4, ease: "easeIn" },
      }}
      className="min-h-[80vh] flex items-center justify-center py-12 xl:py-0"
    >
      <div className="container mx-auto">
        <Tabs
          defaultValue="about"
          className="flex flex-col xl:flex-row gap-[60px]"
        >
          <TabsList className="flex flex-col w-full max-w-[380px] mx-auto xl:mx-0 gap-6">
            <TabsTrigger value="about">About me</TabsTrigger>
            <TabsTrigger value="skills">Skills</TabsTrigger>
            <TabsTrigger value="experience">Experience</TabsTrigger>
            <TabsTrigger value="volunteering">Volunteering</TabsTrigger>
          </TabsList>

          {/* content */}
          <div className="min-h-[70vh] w-full">
            {/* experience */}
            <TabsContent value="experience" className="w-full">
              <div className="flex flex-col gap-[30px] text-center xl:text-left">
                <h3 className="text-4xl font-bold">{experience.title}</h3>
                <p className="max-w-[600px] text-white/60 mx-auto xl:mx-0">
                  {experience.description}
                </p>
                <ScrollArea className="h-[400px]">
                  <ul className="grid grid-cols-1 lg:grid-cols-2 gap-[30px]">
                    {experience.items.map((item, index) => {
                      return (
                        <li
                          key={index}
                          className="bg-[#232329] h-[184px] py-6 px-10 rounded-xl flex flex-col justify-center items-center lg:items-start gap-1"
                        >
                          <span className="text-accent">{item.duration}</span>
                          <h3 className="text-xl max-w-[260px] min-h-[60px] text-center lg:text-left">
                            {item.position}
                          </h3>
                          <div className="flex items-center gap-3">
                            {/* dot */}
                            <span className="w-[6px] h-[6px] rounded-full bg-accent"></span>
                            <p className="text-white/60">{item.company}</p>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </ScrollArea>
              </div>
            </TabsContent>

            {/* skills */}
            <TabsContent value="skills" className="w-full h-full">
              <div className="flex flex-col gap-[30px]">
                <div className="flex flex-col gap-[30px] text-center xl:text-left">
                  <h3 className="text-4xl font-bold">{skills.title}</h3>
                  <p className="max-w-[600px] text-white/60 mx-auto xl:mx-0">
                    {skills.description}
                  </p>
                </div>
                <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 xl:gap-[30px]">
                  {skills.skillList.map((skill, index) => {
                    return (
                      <li key={index}>
                        <TooltipProvider delayDuration={100}>
                          <Tooltip>
                            <TooltipTrigger className="w-full h-[150px] bg-[#232329] rounded-xl flex justify-center items-center group">
                              <div className="text-6xl group-hover:text-accent transition-all duration-300">
                                {skill.icon}
                              </div>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p className="capitalize">{skill.name}</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </TabsContent>

            {/* about */}
            <TabsContent
              value="about"
              className="w-full text-center xl:text-left"
            >
              <div className="flex flex-col gap-[30px]">
                <h3 className="text-4xl font-bold">{about.title}</h3>
                <p className="max-w-[600px] text-white/60 mx-auto xl:mx-0">
                  {about.description}
                </p>
                <ul className="grid grid-cols-1 xl:grid-cols-2 gap-y-6 max-w-[620px] mx-auto xl:mx-0">
                  {about.info.map((item, index) => {
                    return (
                      <li
                        key={index}
                        className="flex items-center justify-center xl:justify-start gap-4"
                      >
                        <span className="text-white/60">{item.fieldName}</span>
                        <span className="text-xl">{item.fieldValue}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </TabsContent>
            {/* volunteering */}
            <TabsContent
              value="volunteering"
              className="w-full text-center xl:text-left"
            >
              <div className="flex flex-col gap-[30px]">
                <h3 className="text-4xl font-bold">{volunteering.title}</h3>
                <a
      href="https://firstracquet.org"
      target="_blank"
      rel="noopener noreferrer"
      className="text-xl font-semibold hover:underline"
    ><h2 className="text-xl font-semibold">First Racquet | Executive VP of Directors</h2></a>
                <p className="max-w-[600px] text-white/60 mx-auto xl:mx-0">
                  {volunteering.description}
                </p>
              </div>
            </TabsContent>
            
          </div>
        </Tabs>
      </div>
    </motion.div>
  );
};

export default Resume;
