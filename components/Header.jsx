import Link from "next/link";
import { Button } from "./ui/button";

// components
import Nav from "./Nav";
import MobileNav from "./MobileNav";

const Header = () => {
  return (
    <>
      <link rel="icon" href="/icon.ico" />
      <header className="py-8 xl:py-12 text-white">
        <div className="container mx-auto flex justify-between items-center">
          {/* logo */}
          <Link href="/" legacyBehavior>
            <a className="flex items-center gap-2">
              <img src="/icon.ico" alt="icon" className="w-10 h-10" />
              <h1 className="text-4xl font-semibold flex items-center">
                Kacem<span className="text-accent">.</span>
              </h1>
            </a>
          </Link>

          {/* desktop nav & hire me button */}
          <div className="hidden xl:flex items-center gap-8">
            <Nav />
            <Link href="/contact" legacyBehavior>
              <Button>Hire me</Button>
            </Link>
          </div>

          {/* mobile nav */}
          <div className="xl:hidden">
            <MobileNav />
          </div>
        </div>
      </header>
    </>
  );
};

export default Header;
