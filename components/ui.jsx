"use client";

import { AvatarImage, AvatarFallback, Avatar } from "/components/ui/avatar";
import { Input } from "/components/ui/input";
import { Button } from "/components/ui/button";
import Countdown from "/components/ui/countdown";
import { useChat } from "ai/react";
import { useState } from "react";
import SelfieUI from "/components/selfieui";

export function UI() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "/api/chat",
  });

  //state hook for when the user clicks the button to take a selfie
  const [selfie, setSelfie] = useState(false);
  const [concepts, setConcepts] = useState(null);
  const [image, setImage] = useState(null);

  return (
    <div className="flex flex-col h-screen">
      <header className="flex items-center justify-between px-4 py-2 border-b">
        <h1 className="text-lg font-semibold">Longevity Copilot</h1>
        <Avatar>
          <AvatarImage alt="@shadcn" src="/placeholder-user.jpg" />
          <AvatarFallback>AC</AvatarFallback>
        </Avatar>
      </header>
      {/*<main className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, index) => (
          <div key={index} className="flex items-end space-x-2">
            <Avatar className="w-10 h-10">
              <AvatarImage alt="@shadcn" src="/placeholder-user.jpg" />
              <AvatarFallback>AC</AvatarFallback>
            </Avatar>
            <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800">
              <p className="text-sm">{m.content}</p>
            </div>
          </div>
        ))}
      </main>
      <div className="p-4 border-t">
        <form className="flex items-center space-x-2" onSubmit={handleSubmit}>
          <Input
            className="flex-1"
            placeholder="Type a message"
            value={input}
            onChange={handleInputChange}
          />
          <Button size="icon" variant="outline">
            <SmileIcon className="h-6 w-6" />
          </Button>
          <Button size="icon" type="submit">
            <SendIcon className="h-6 w-6" />
          </Button>
        </form>
      </div>*/}
      <section className="flex flex-col items-center p-4 border-t">
        <h2 className="text-lg font-semibold mb-2">Next Selfie in:</h2>
        <Countdown />
        <Button className="mt-4" onClick={() => setSelfie(true)}>
          Take Selfie Now
        </Button>
        {selfie && (
          <SelfieUI
            setSelfie={setSelfie}
            setConcepts={setConcepts}
            setImage={setImage}
          />
        )}
      </section>
      <section className="flex flex-row items-center p-4 border-t">
        <div className="w-1/2">
          {image && <img src={image} alt="Selfie" className="rounded-full " />}
        </div>
        <div className="w-1/2">
          {image && (
            <h2 className="text-lg font-semibold mb-2">Predicted Age:</h2>
          )}
          {concepts &&
            concepts.map((concept, index) => (
              <div
                key={index}
                className="bg-blue-200 rounded-full px-3 py-1 text-sm font-semibold text-gray-700 mr-2 mb-2"
              >
                {concept.name}: {concept.value}
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}

function SmileIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M8 14s1.5 2 4 4-2" />
      <line x1="9" x2="9.01" y1="9" y2="9" />
      <line x1="15" x2="15.01" y1="9" y2="9" />
    </svg>
  );
}

function SendIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m22 2-7 20-4-9-9-4Z" />
      <path d="M22 2 11 13" />
    </svg>
  );
}
