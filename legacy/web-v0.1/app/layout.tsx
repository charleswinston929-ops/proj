import "./globals.css";
import Link from "next/link";

export const metadata={title:"Sports Intelligence",description:"Football forecasting and model research"};

export default function RootLayout({children}:{children:React.ReactNode}){
 return <><nav className="nav"><Link href="/"><strong>SPORTS INTELLIGENCE</strong></Link><div className="navlinks"><Link href="/matches">Matches</Link><Link href="/predictions">Predictions</Link><Link href="/models">Models</Link><Link href="/research">Research</Link></div></nav>{children}</>;
}
