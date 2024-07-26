"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { FaPhoneAlt, FaEnvelope, FaMapMarkerAlt } from "react-icons/fa";

const info = [
  {
    icon: <FaPhoneAlt />,
    title: "Phone",
    description: "(+1) 832 410 2200",
  },
  {
    icon: <FaEnvelope />,
    title: "Email",
    description: "101kacem101@gmail.com",
  },
  {
    icon: <FaMapMarkerAlt />,
    title: "Affiliation",
    description: "Shadow Creek High School",
  },
];

import { motion } from "framer-motion";

const Contact = () => {
  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{
        opacity: 1,
        transition: { delay: 2.4, duration: 0.4, ease: "easeIn" },
      }}
      className="py-6"
    >
      <div className="container mx-auto">
        <div className="flex flex-col xl:flex-row gap-[30px]">
          {/* form */}
          <div className="xl:w-[54%] order-2 xl:order-none">
          <form className="flex flex-col gap-6 p-10 bg-[#27272c] rounded-xl" action="https://formspree.io/f/mnqeypbd"  method="POST">
          <h3 className="text-4xl text-accent">Let's work together. Reach Out!</h3>
          <p className="text-white/60"></p>
          
          {/* input */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input name="firstname" placeholder="First name" required />
            <Input name="lastname" placeholder="Last name" required />
            <Input type="email" name="email" placeholder="Email address" required />
            <Input type="tel" name="phone" placeholder="Phone number" />
          </div>
          
          {/* select */}
          <Select name="service" required>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a service" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Select a service</SelectLabel>
                <SelectItem value="N/A">N/A</SelectItem>
                <SelectItem value="web-development">Web Development</SelectItem>
                <SelectItem value="ui-ux-design">UI/UX Design</SelectItem>
                <SelectItem value="ios-app-development">iOS App Development</SelectItem>
                <SelectItem value="frontend-development">Frontend Development</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          
          {/* textarea */}
          <Textarea
            name="message"
            className="h-[200px]"
            placeholder="Type your message here."
            required
          />
          
          {/* btn */}
          <Button type="submit" size="md" className="max-w-40">
            Send message
          </Button>
        </form>
          </div>
          {/* info */}
          <div className="flex-1 flex items-center xl:justify-end order-1 xl:order-none mb-8 xl:mb-0">
            <ul className="flex flex-col gap-10">
              {info.map((item, index) => {
                return (
                  <li key={index} className="flex items-center gap-6">
                    <div className="w-[52px] h-[52px] xl:w-[72px] xl:h-[72px] bg-[#27272c] text-accent rounded-md flex items-center justify-center">
                      <div className="text-[28px]">{item.icon}</div>
                    </div>
                    <div className="flex-1">
                      <p className="text-white/60">{item.title}</p>
                      <h3 className="text-xl">{item.description}</h3>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      </div>
    </motion.section>
  );
};

export default Contact;
