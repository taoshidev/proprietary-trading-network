"use client";

import {useState, useMemo, useEffect} from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Button } from "@/components/ui/button";

import WatchForm from "@/features/watch/components/watch-form";

import { useToast } from "@/hooks/use-toast"

import useSignalStore from "@/app/store/signalStore";

import { saveConfig, getConfig } from "@/app/actions/config";
import { type Config } from '@/types'

export default function Watch() {
  const { toast } = useToast();
  const { signals, isWatching, startWebSocket, stopWebSocket } = useSignalStore();
  const [editMode, setEditMode] = useState(false);
  const [data, setData] = useState<Config>();

  const handleSubmit = async (data: Config): Promise<void> => {
    const response = await saveConfig(data);

    if (response.data) {
      setEditMode(false);
      setData(response.data);

      toast({
        title: "Success!",
        description: response.message,
      });
    }
  };

  const toggleEdit = () => {
    setEditMode((prev) => !prev);
  }

  const showEditMode = useMemo(() => {
    return editMode || !data;
  }, [editMode, data]);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const config = await getConfig();
        setData(config.data);
      } catch(error) {
        console.log("error", error);
      }
    };

    fetchConfig()
  }, [])

  return (
    <>
      <Card className='w-full'>
        <CardHeader>
          <CardTitle>Bybit</CardTitle>
          <CardDescription>{showEditMode ? 'Add your Bybit account' : 'Bybit account configured'}</CardDescription>
        </CardHeader>
        <CardContent>
          {showEditMode ? (
            <WatchForm onSubmitAction={handleSubmit} data={data} onCancelAction={toggleEdit}/>
          ) : (
            <div>
              <div className='w-full text-center py-20'>
                <Button variant='link' onClick={toggleEdit}>Edit Connection</Button>
              </div>
              <div className='flex w-full gap-8'>
                <Button className='w-full' onClick={startWebSocket} disabled={isWatching}>
                  {isWatching ? "Watching..." : "Start Watching"}
                </Button>
                <Button className='w-full' onClick={stopWebSocket} disabled={!isWatching} variant="secondary">
                  Stop Watching
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Sheet>
        <SheetTrigger className='text-xs text-center w-full mt-4 hover:underline hover:cursor-pointer'>View Recent
          Signals</SheetTrigger>
        <SheetContent>
          <SheetHeader>
            <SheetTitle className='mb-6'>Recent Signals</SheetTitle>
            <div>
              {signals.map((signal, index) => (
                <Card key={index} className='mb-4'>
                  <CardHeader>
                    <CardTitle>Signal sent to PTN </CardTitle>
                    <CardDescription>{signal.orderType}:{signal.trade_pair.trade_pair} Lvg: {signal.leverage}</CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    </>
  );
}
