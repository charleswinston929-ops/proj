import MatchClient from "../../../components/MatchClient";
export default async function MatchPage({params}:{params:Promise<{matchId:string}>}){const {matchId}=await params;return <MatchClient matchId={matchId}/>;}
