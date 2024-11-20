"use client";

import {useEffect} from "react";

import { last } from 'lodash';

import { useToast } from "@/hooks/use-toast"
import { Signal } from '@/types';

import useSignalStore from "@/app/store/signalStore";

import Start from "@/features/start/components/start";
import Watch from "@/features/watch/components/watch";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Home() {
  const { toast } = useToast();
  const { signals } = useSignalStore();

  const toastDesc = (signal: Signal) => {
    return `${signal.orderType}:${signal.trade_pair.trade_pair} Lvg: ${signal.leverage}`;
  };

  useEffect(() => {
    if (signals.length) {

      toast({
        title: 'Sent to PTN',
        description: toastDesc(last(signals) as Signal),
      });
    }
  }, [toast, signals]);

  return (
    <div className="flex flex-col items-center justify-items-center -8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex gap-8 row-start-2 items-center sm:items-start">
        <div className='w-[500px]'>
          <div className='text-center my-20 font-edu text-md text-[#EA4436]'>easy miner</div>

          <Tabs defaultValue="miner" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="miner">Start Miner</TabsTrigger>
              <TabsTrigger value="watch">Watch Signals</TabsTrigger>
            </TabsList>
            <TabsContent value="miner">
              <Start />
            </TabsContent>
            <TabsContent value="watch">
              <Watch />
            </TabsContent>
          </Tabs>

        </div>
      </main>
    </div>
  );
}
