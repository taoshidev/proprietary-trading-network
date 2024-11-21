"use client";

import { useState, useEffect, useRef } from "react";

import useLogStore from "@/app/store/logStore";
import { useToast } from "@/hooks/use-toast";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";

import StartForm from "@/features/start/components/start-form";

import { startMiner, startServer } from "@/app/actions/miner";

export default function Start() {
  const initialize = useLogStore((state) => state.initialize);
  const logs = useLogStore((state) => state.logs);

  const { toast } = useToast();
  const [editMode, setEditMode] = useState(false);
  const [data, setData] = useState(null);
  const [logsModal, showLogsModal] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (data): Promise<void> => {
    await startMiner(data);
    //await startServer();

    // if (response.data) {
    //   setEditMode(false);
    //   setData(response.data);
    //
    //   toast({
    //     title: "Success!",
    //     description: response.message,
    //   });
    // }

    showLogsModal(true);
  };

  // useEffect(() => {
  //   initialize();
  // }, [initialize]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const openClose = () => {
    showLogsModal((state) => !state);
  };

  return (
    <>
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Start Your Miner</CardTitle>
          <CardDescription>Configure and Start Your Miner</CardDescription>
        </CardHeader>
        <CardContent>
          <StartForm onSubmitAction={handleSubmit} />
        </CardContent>
      </Card>
      <div
        className="text-xs text-center w-full mt-4 hover:underline hover:cursor-pointer"
        onClick={openClose}
      >
        View Logs
      </div>
      <Dialog open={logsModal} onOpenChange={openClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Miner Logs</DialogTitle>
            <DialogDescription>View Your Miner Logs</DialogDescription>
            <ScrollArea
              ref={scrollRef}
              className="h-[70vh] w-full rounded-md border p-4"
            >
              <div>
                {logs.map((log, index) => (
                  <pre className="text-xs" key={index}>
                    {log}
                  </pre>
                ))}
              </div>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </>
  );
}
